#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <pthread.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdatomic.h>

#define N_TASKS 128
#define MAX_DIFFICULTY_LEVEL 10
#define REQUEST_TAG 0
#define TASK_TAG 1
#define RESULT_TAG 2
#define ITERATIONS 5

typedef struct
{
    int task_id;
    int difficulty;
    double result;
    int wasTransferred;
} Task;

typedef struct
{
    Task *tasks;
    int count;
    pthread_mutex_t mutex;
} TaskQueue;

typedef struct
{
    int rank;
    int initial_weight;
    int completed_tasks;
    int total_weight;
    int own_tasks;
    double work_time;
} ProcessInfo;

typedef struct
{
    int rank;
    int size;
    ProcessInfo *process_info;
    MPI_Datatype mpi_task_type;
    MPI_Datatype mpi_process_type;
    double local_result;
    pthread_mutex_t result_mutex;
} ThreadArgs;

typedef struct
{
    int *counts;
    int *displacements;
    TaskQueue global_queue;
    Task *original_tasks;
    ProcessInfo *global_process_infos;
    ProcessInfo *iteration_process_infos;
    ProcessInfo *current_process_info;
} ProgramState;

TaskQueue local_queue;

MPI_Datatype create_task_type()
{
    MPI_Datatype task_type;
    int block_lengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT};

    displacements[0] = offsetof(Task, task_id);
    displacements[1] = offsetof(Task, difficulty);
    displacements[2] = offsetof(Task, result);
    displacements[3] = offsetof(Task, wasTransferred);

    MPI_Type_create_struct(4, block_lengths, displacements, types, &task_type);
    MPI_Type_commit(&task_type);

    return task_type;
}

MPI_Datatype create_process_info_type()
{
    MPI_Datatype process_info_type;
    int block_lengths[6] = {1, 1, 1, 1, 1, 1};
    MPI_Aint displacements[6];
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};

    displacements[0] = offsetof(ProcessInfo, rank);
    displacements[1] = offsetof(ProcessInfo, initial_weight);
    displacements[2] = offsetof(ProcessInfo, completed_tasks);
    displacements[3] = offsetof(ProcessInfo, total_weight);
    displacements[4] = offsetof(ProcessInfo, own_tasks);
    displacements[5] = offsetof(ProcessInfo, work_time);

    MPI_Type_create_struct(6, block_lengths, displacements, types, &process_info_type);
    MPI_Type_commit(&process_info_type);

    return process_info_type;
}

void add_task(TaskQueue *q, Task task)
{
    pthread_mutex_lock(&q->mutex);
    if (q->count < N_TASKS)
    {
        q->tasks[q->count++] = task;
    }
    else
    {
        fprintf(stderr, "Queue overflow!\n");
    }
    pthread_mutex_unlock(&q->mutex);
}

int get_task(TaskQueue *q, Task *task)
{
    pthread_mutex_lock(&q->mutex);
    if (q->count == 0)
    {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }
    *task = q->tasks[--q->count];
    pthread_mutex_unlock(&q->mutex);
    return 1;
}




void uniformDistribution(int *numberOfTasksForProcesses, int size)
{
    int perProcess = N_TASKS / size;
    for (int i = 0; i < size; i++)
    {
        numberOfTasksForProcesses[i] = perProcess;
    }
}

void skewedLeftDistribution(int *numberOfTasksForProcesses, int size)
{
    double weights[size];
    double total_weight = 0.0;

    for (int i = 0; i < size; i++) {
        weights[i] = (double)(size - i); // убывающее: [size, size-1, ..., 1]
        total_weight += weights[i];
    }

    int total_assigned = 0;
    for (int i = 0; i < size; i++) {
        numberOfTasksForProcesses[i] = (int)((weights[i] / total_weight) * N_TASKS);
        total_assigned += numberOfTasksForProcesses[i];
    }

    // Распределяем оставшиеся задачи
    int remaining = N_TASKS - total_assigned;
    for (int i = 0; remaining > 0; i = (i + 1) % size) {
        numberOfTasksForProcesses[i]++;
        remaining--;
    }
}




void triangularDistribution(int *numberOfTasksForProcesses, int size)
{
    int total = 0;
    for (int i = 0; i < size; i++)
    {
        int val = (i <= size / 2) ? (i + 1) : (size - i);
        numberOfTasksForProcesses[i] = val;
        total += val;
    }

    double scale = (double)N_TASKS / total;
    for (int i = 0; i < size; i++)
        numberOfTasksForProcesses[i] = (int)(numberOfTasksForProcesses[i] * scale);

    int remaining = N_TASKS;
    for (int i = 0; i < size; i++) remaining -= numberOfTasksForProcesses[i];
    for (int i = 0; remaining > 0; i = (i + 1) % size)
    {
        numberOfTasksForProcesses[i]++;
        remaining--;
    }
}

void singleHeavyProcessDistribution(int *numberOfTasksForProcesses, int size)
{
    for (int i = 0; i < size; i++)
        numberOfTasksForProcesses[i] = 0;

    int center = size / 2;
    numberOfTasksForProcesses[center] = N_TASKS;
}


void distributeTasks(int distributionCode, int *numberOfTasksForProcesses, int size)
{
    if (size == 1)
    {
        numberOfTasksForProcesses[0] = N_TASKS;
        return;
    }

    switch (distributionCode)
    {
    case 0:
        uniformDistribution(numberOfTasksForProcesses, size);
        break;
    case 1:
        skewedLeftDistribution(numberOfTasksForProcesses, size); 
        break;
    case 2:
        triangularDistribution(numberOfTasksForProcesses, size);
        break;
    case 3:
        singleHeavyProcessDistribution(numberOfTasksForProcesses, size);
        break;
    }
}



double exp_taylor(int iterations)
{
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i < iterations * 1000000; i++)
    {
        term *= (1.0 / i);
        result += term;
    }
    return result;
}

void update_process_stats(ThreadArgs *targs, Task *task,
                          int *completed_tasks, int *total_weight,
                          int *own_tasks, double *calc_time)
{
    double start = MPI_Wtime();
    task->result = exp_taylor(task->difficulty);
    *calc_time += MPI_Wtime() - start;

    pthread_mutex_lock(&targs->result_mutex);
    targs->local_result += task->result;
    pthread_mutex_unlock(&targs->result_mutex);

    (*completed_tasks)++;
    *total_weight += task->difficulty;
    if (!task->wasTransferred)
        (*own_tasks)++;
}

int process_local_tasks(ThreadArgs *targs, Task *task,
                        int *completed_tasks, int *total_weight,
                        int *own_tasks, double *calc_time)
{
    if (get_task(&local_queue, task))
    {
        update_process_stats(targs, task, completed_tasks, total_weight, own_tasks, calc_time);
        return 1;
    }
    return 0;
}

int request_tasks_from_neighbors(int rank, int size, MPI_Datatype task_type)
{
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    MPI_Request req[2];

    // избежание дедлока
    MPI_Isend(&rank, 1, MPI_INT, left, REQUEST_TAG, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(&rank, 1, MPI_INT, right, REQUEST_TAG, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    Task received_task;
    int received = 0;
    MPI_Status status;

    for (int i = 0; i < 2; i++)
    {
        MPI_Recv(&received_task, 1, task_type, MPI_ANY  _SOURCE, TASK_TAG, MPI_COMM_WORLD, &status);
        if (received_task.task_id != -1)
        {
            add_task(&local_queue, received_task);
            received = 1;
        }
    }
    return received;
}

int request_tasks_from_all(int rank, int size, MPI_Datatype task_type)
{
    Task received_task;
    MPI_Status status;
    int received = 0;

    for (int i = 0; i < size; i++)
    {
        if (i == rank)
            continue;

        MPI_Send(&rank, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
        MPI_Recv(&received_task, 1, task_type, i, TASK_TAG, MPI_COMM_WORLD, &status);

        if (received_task.task_id != -1)
        {
            add_task(&local_queue, received_task);
            received = 1;
            break;
        }
    }
    return received;
}

int should_terminate(int size)
{
    int local_count;
    pthread_mutex_lock(&local_queue.mutex); 
    local_count = local_queue.count;

    if (size == 1){
        bool c = local_count == 0;
        pthread_mutex_unlock(&local_queue.mutex);
        return c;
    }

    int global_count;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    bool c = global_count == 0;
    pthread_mutex_unlock(&local_queue.mutex);
    return c;
}

void *worker_thread(void *args)
{
    ThreadArgs *targs = (ThreadArgs *)args;
    Task task;
    int own_tasks = 0, completed_tasks = 0, total_weight = 0;
    double calc_time = 0.0;

    for (int i = 0; i < local_queue.count; i++)
    {
        total_weight += local_queue.tasks[i].difficulty;
    }
    int initial_weight = total_weight;
    total_weight = 0;

    while (1)
    {
        if (process_local_tasks(targs, &task, &completed_tasks, &total_weight, &own_tasks, &calc_time))
            continue;

        if (targs->size > 1)
        {
            if (request_tasks_from_neighbors(targs->rank, targs->size, targs->mpi_task_type))
                continue;

            if (request_tasks_from_all(targs->rank, targs->size, targs->mpi_task_type))
                continue;
        }

        if (should_terminate(targs->size))
            break;
    }

    targs->process_info->rank = targs->rank;
    targs->process_info->initial_weight = initial_weight;
    targs->process_info->completed_tasks = completed_tasks;
    targs->process_info->total_weight = total_weight;
    targs->process_info->own_tasks = own_tasks;
    targs->process_info->work_time = calc_time;

    if (targs->size > 1)
    {
        int stop_signal = -1;
        MPI_Send(&stop_signal, 1, MPI_INT, targs->rank, REQUEST_TAG, MPI_COMM_WORLD);
    }

    return NULL;
}

int handle_task_request(ThreadArgs *targs, int source_rank)
{
    Task task_to_send;

    if (get_task(&local_queue, &task_to_send))
    {
        task_to_send.wasTransferred = 1;
        MPI_Send(&task_to_send, 1, targs->mpi_task_type,
                 source_rank, TASK_TAG, MPI_COMM_WORLD);
        return 1;
    }

    Task empty_task = {.task_id = -1};
    MPI_Send(&empty_task, 1, targs->mpi_task_type,
             source_rank, TASK_TAG, MPI_COMM_WORLD);
    return 0;
}

void *server_thread(void *args)
{
    ThreadArgs *targs = (ThreadArgs *)args;
    int rank = targs->rank;
    MPI_Status status;

    while (1)
    {
        int request;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_SOURCE == rank && request == -1)
            break;

        handle_task_request(targs, status.MPI_SOURCE);
    }

    return NULL;
}




void check_allocation(void *ptr, const char *msg, int mpi_err_code)
{
    if (!ptr)
    {
        fprintf(stderr, "Allocation failed: %s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, mpi_err_code);
    }
}

void run_worker_and_server(ThreadArgs *args, int size)
{
    pthread_t worker;
    pthread_create(&worker, NULL, worker_thread, args);

    if (size > 1)
    {
        server_thread(args);
    }

    pthread_join(worker, NULL);
}

void initialize_mpi_environment(int *rank, int *size,
                                MPI_Datatype *mpi_task_type,
                                MPI_Datatype *mpi_process_type)
{
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        printf("MPI_THREAD_MULTIPLE not supported!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);

    *mpi_task_type = create_task_type();
    *mpi_process_type = create_process_info_type();
    pthread_mutex_init(&local_queue.mutex, NULL);
}

void initialize_global_queue(TaskQueue *queue)
{
    queue->tasks = malloc(N_TASKS * sizeof(Task));
    check_allocation(queue->tasks, "global_queue.tasks", 3);

    queue->count = 0;
    for (int i = 0; i < N_TASKS; i++)
    {
        Task t = {
            .task_id = i,
            .difficulty = (i * i * i) % 10 + 1,
            .wasTransferred = 0};
        add_task(queue, t);
    }
}

void calculate_distribution(int dist_code, int size,
                            int *counts, int *displacements)
{
    int *tasks_per_process = malloc(size * sizeof(int));
    check_allocation(tasks_per_process, "tasks_per_process", 2);

    distributeTasks(dist_code, tasks_per_process, size);

    displacements[0] = 0;
    counts[0] = tasks_per_process[0];
    for (int i = 1; i < size; i++)
    {
        counts[i] = tasks_per_process[i];
        displacements[i] = displacements[i - 1] + counts[i - 1];
    }

    free(tasks_per_process);
}

void initialize_local_resources(ProgramState *state, int rank, int size)
{
    state->current_process_info = malloc(sizeof(ProcessInfo));
    check_allocation(state->current_process_info, "current_process_info", 5);

    local_queue.tasks = malloc((state->counts[rank] > 0 ? state->counts[rank] : 1) * sizeof(Task));
    check_allocation(local_queue.tasks, "local_queue.tasks", 4);

    if (rank == 0)
    {
        state->global_process_infos = calloc(size, sizeof(ProcessInfo));
        state->iteration_process_infos = malloc(size * sizeof(ProcessInfo));
        check_allocation(state->global_process_infos, "global_process_infos", 6);
        check_allocation(state->iteration_process_infos, "iteration_process_infos", 6);
    }
}

ThreadArgs create_thread_args(int rank, int size,
                              MPI_Datatype *mpi_task_type,
                              MPI_Datatype *mpi_process_type)
{
    ThreadArgs args = {
        .rank = rank,
        .size = size,
        .process_info = NULL,
        .mpi_task_type = *mpi_task_type,
        .mpi_process_type = *mpi_process_type,
        .local_result = 0.0,
        .result_mutex = PTHREAD_MUTEX_INITIALIZER};
    return args;
}


void execute_iterations(int iterations, int rank, int size,
                        ProgramState *state, ThreadArgs *thread_args)
{
    thread_args->process_info = state->current_process_info;

    for (int j = 0; j < iterations; j++)
    {
        local_queue.count = state->counts[rank];
        memset(state->current_process_info, 0, sizeof(ProcessInfo));
        thread_args->local_result = 0.0;

        if (rank == 0)
        {
            memcpy(state->global_queue.tasks, state->original_tasks,
                   N_TASKS * sizeof(Task));
        }

        MPI_Scatterv(state->global_queue.tasks, state->counts, state->displacements,
                     thread_args->mpi_task_type, local_queue.tasks,
                     state->counts[rank], thread_args->mpi_task_type,
                     0, MPI_COMM_WORLD);

        run_worker_and_server(thread_args, size);

        double global_result = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&thread_args->local_result, &global_result, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Gather(thread_args->process_info, 1, thread_args->mpi_process_type,
                   (rank == 0) ? state->iteration_process_infos : NULL,
                   (rank == 0) ? 1 : 0, thread_args->mpi_process_type,
                   0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 0; i < size; i++)
            {
                state->global_process_infos[i].rank = state->iteration_process_infos[i].rank;
                state->global_process_infos[i].initial_weight += state->iteration_process_infos[i].initial_weight;
                state->global_process_infos[i].total_weight += state->iteration_process_infos[i].total_weight;
                state->global_process_infos[i].completed_tasks += state->iteration_process_infos[i].completed_tasks;
                state->global_process_infos[i].own_tasks += state->iteration_process_infos[i].own_tasks;
                state->global_process_infos[i].work_time += state->iteration_process_infos[i].work_time;
            }
        }
    }
}


void print_process_stats(const ProcessInfo *info)
{
    printf("-------------------------------------------------\n");
    printf("Process %d:\n", info->rank);
    printf("Initial total weight: %d\n", info->initial_weight);
    printf("Total weight of completed tasks: %d\n", info->total_weight);
    printf("Completed tasks: %d\n", info->completed_tasks);

    double own_tasks_percent = info->completed_tasks == 0 ? 0 : ((double)info->own_tasks / info->completed_tasks * 100);
    printf("Percentage of own tasks: %.2f%%\n", own_tasks_percent);

    printf("Useful working time: %.3f sec\n", info->work_time);
    printf("-------------------------------------------------\n\n");
}

void print_statistics(double total_time, ProgramState *state, int size)
{
    printf("Time: %.3f\n", total_time);

    double max_time = 0, min_time = total_time;
    for (int i = 0; i < size; i++)
    {
        if (state->global_process_infos[i].work_time > max_time)
        {
            max_time = state->global_process_infos[i].work_time;
        }
        if (state->global_process_infos[i].work_time < min_time)
        {
            min_time = state->global_process_infos[i].work_time;
        }
    }

    printf("Load imbalance: %.3f\n\n", max_time / min_time);
    for (int i = 0; i < size; i++)
    {
        print_process_stats(&state->global_process_infos[i]);
    }
}


void cleanup_resources(ProgramState *state, int rank,
                       MPI_Datatype *mpi_task_type,
                       MPI_Datatype *mpi_process_type)
{
    free(state->current_process_info);
    free(state->counts);
    free(state->displacements);
    free(local_queue.tasks);
    pthread_mutex_destroy(&local_queue.mutex);

    if (rank == 0)
    {
        free(state->global_process_infos);
        free(state->iteration_process_infos);
        free(state->original_tasks);
        free(state->global_queue.tasks);
    }

    MPI_Type_free(mpi_task_type);
    MPI_Type_free(mpi_process_type);
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <distribution_code>\n", argv[0]);
        return 0;
    }

    int distribution_code = atoi(argv[1]);
    if (distribution_code < 0 || distribution_code > 5)
    {
        printf("Invalid distribution code!\n");
        return 0;
    }

    int rank, size;
    MPI_Datatype mpi_task_type, mpi_process_type;
    initialize_mpi_environment(&rank, &size, &mpi_task_type, &mpi_process_type);

    ProgramState state = {0};
    state.counts = malloc(size * sizeof(int));
    state.displacements = malloc(size * sizeof(int));
    check_allocation(state.counts, "counts", 2);
    check_allocation(state.displacements, "displacements", 2);

    if (rank == 0)
    {
        initialize_global_queue(&state.global_queue);
        calculate_distribution(distribution_code, size, state.counts, state.displacements);

        state.original_tasks = malloc(N_TASKS * sizeof(Task));
        check_allocation(state.original_tasks, "original_tasks", 3);
        memcpy(state.original_tasks, state.global_queue.tasks, N_TASKS * sizeof(Task));
    }

    MPI_Bcast(state.counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(state.displacements, size, MPI_INT, 0, MPI_COMM_WORLD);

    initialize_local_resources(&state, rank, size);
    ThreadArgs thread_args = create_thread_args(rank, size, &mpi_task_type, &mpi_process_type);

    double start_time = MPI_Wtime();
    execute_iterations(ITERATIONS, rank, size, &state, &thread_args);
    double total_time = MPI_Wtime() - start_time;

    if (rank == 0)
    {
        print_statistics(total_time, &state, size);
    }

    cleanup_resources(&state, rank, &mpi_task_type, &mpi_process_type);
    MPI_Finalize();
    return 0;
}
