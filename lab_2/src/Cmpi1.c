#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const double eps = 0.00001;
const int maxIterationsCount = 5000;
const double taul = 0.00001;

void createMatrixPartForProcesses(int* countOfLinesForProcess, int* offsetArray, int* countOfMatrElem_s, int* countOfSkipedElem, int N, int size) {
    int currentOffset = 0;
    for (int i = 0; i < size; ++i) {
        countOfLinesForProcess[i] = N / size;
        if (i < N % size) {
            ++countOfLinesForProcess[i];
        }
        offsetArray[i] = currentOffset;
        currentOffset += countOfLinesForProcess[i];
        countOfMatrElem_s[i] = countOfLinesForProcess[i] * N;
        countOfSkipedElem[i] = offsetArray[i] * N;
    }
}

double countAbsSqr(const double* vec, const int N) {
    double absSqr = 0.0;
    for (int i = 0; i < N; ++i) {
        absSqr += pow(vec[i], 2);
    }
    return absSqr;
}

void firstInit(double* A, double* b, double* x, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 1.0;
        }
        A[i * N + i] = 2.0;
        x[i] = 0.0;
        b[i] = N + 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        if (argc == 1) {
            printf("Error: N is required as a command-line argument!\n");
        } else {
            printf("Error: too many arguments!\n");
        }
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Error: N must be a positive integer.\n");
        return 1;
    }

    MPI_Init(&argc, &argv);
    double startTime = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* countLinesForProc = (int*)malloc(size * sizeof(int));
    int* offsetArray = (int*)malloc(size * sizeof(int));
    int* countOfMatrElem_s = (int*)malloc(size * sizeof(int));
    int* countOfSkipedElem = (int*)malloc(size * sizeof(int));
    createMatrixPartForProcesses(countLinesForProc, offsetArray, countOfMatrElem_s, countOfSkipedElem, N, size);

    double* partOfA = (double*)malloc(countLinesForProc[rank] * N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));

    double g_x = 0.0;
    double abs_b = 0.0;
    if (rank == 0) {
        firstInit(A, b, x, N);
        g_x = 1;
    }

    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    abs_b = sqrt(countAbsSqr(b, N));

    MPI_Scatterv(A, countOfMatrElem_s, countOfSkipedElem, MPI_DOUBLE, partOfA, countOfMatrElem_s[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* Ax_b = (double*)malloc(countLinesForProc[rank] * sizeof(double));
    double* partOfX = (double*)malloc(countLinesForProc[rank] * sizeof(double));

    int iterationCount = 0;

    while (g_x > eps && iterationCount < maxIterationsCount) {
        for (int i = 0; i < countLinesForProc[rank]; ++i) {
            Ax_b[i] = 0;
            for (int j = 0; j < N; ++j) {
                Ax_b[i] += partOfA[i * N + j] * x[j];
            }
            Ax_b[i] = Ax_b[i] - b[offsetArray[rank] + i];
        }

        for (int i = 0; i < countLinesForProc[rank]; ++i) {
            partOfX[i] = x[offsetArray[rank] + i] - taul * Ax_b[i];
        }

        MPI_Allgatherv(partOfX, countLinesForProc[rank], MPI_DOUBLE, x, countLinesForProc, offsetArray, MPI_DOUBLE, MPI_COMM_WORLD);

        double partAbs = countAbsSqr(Ax_b, countLinesForProc[rank]);
        MPI_Allreduce(&partAbs, &g_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            iterationCount++;
        }

        MPI_Bcast(&iterationCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        g_x = sqrt(g_x) / abs_b;
    }

    if (rank == 0) {
        if (iterationCount < maxIterationsCount) {
            double endTime = MPI_Wtime();
            printf("time: %f sec\n", endTime - startTime);
        } else {
            printf("No converge, change taul sign\n");
        }

        double maxDifference = 0.0;
        for (int i = 0; i < N; ++i) {
            double difference = fabs(x[i] - b[i]);
            if (difference > maxDifference) {
                maxDifference = difference;
            }
        }
        printf("max difference: %lf\n", maxDifference);
    }

    free(countLinesForProc);
    free(offsetArray);
    free(x);
    free(b);
    free(A);
    free(partOfA);
    free(Ax_b);
    free(partOfX);

    MPI_Finalize();

    return 0;
}