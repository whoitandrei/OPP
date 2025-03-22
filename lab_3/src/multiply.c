#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define n1 2500
#define n2 2500
#define n3 2500


typedef struct {
    MPI_Comm grid;
    MPI_Comm row;
    MPI_Comm col;
    int coords[2];
    int dims[2];
    int rank;
    int size;
} GridInfo;

typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

void print_matrix(const Matrix* mat) {
    for(int x = 0; x < mat->rows; ++x) {
        for(int y = 0; y < mat->cols; ++y) {
            printf("%lf ", mat->data[x * mat->cols + y]);
        }
        printf("\n");
    }
    printf("\n");
}

void setup_mpi_environment(GridInfo* grid, int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &grid->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->rank);

    grid->dims[0] = grid->dims[1] = 0;
    if(argc == 3) {
        grid->dims[0] = atoi(argv[1]);
        grid->dims[1] = atoi(argv[2]);
    } else {
        MPI_Dims_create(grid->size, 2, grid->dims);
    }
    
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid->dims, periods, 0, &grid->grid);
    MPI_Cart_coords(grid->grid, grid->rank, 2, grid->coords);
    
    int sub_dims[2];
    sub_dims[0] = 0; sub_dims[1] = 1;
    MPI_Cart_sub(grid->grid, sub_dims, &grid->row);
    sub_dims[0] = 1; sub_dims[1] = 0;
    MPI_Cart_sub(grid->grid, sub_dims, &grid->col);
}

void validate_dimensions(const GridInfo* grid) {
    if((n1 % grid->dims[0] != 0) || (n3 % grid->dims[1] != 0)) {
        if(grid->rank == 0) {
            fprintf(stderr, "Error: n1 and n3 must be divisible by grid dimensions\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
}

void distribute_matrix_a(const GridInfo* grid, Matrix* a, Matrix* sub_a) {
    int sub_rows = n1 / grid->dims[0];
    MPI_Datatype sub_a_type;
    MPI_Type_contiguous(sub_rows * n2, MPI_DOUBLE, &sub_a_type);
    MPI_Type_commit(&sub_a_type);

    if(grid->coords[1] == 0) {
        MPI_Scatter(a->data, 1, sub_a_type, 
                   sub_a->data, 1, sub_a_type, 
                   0, grid->col);
    }
    MPI_Bcast(sub_a->data, 1, sub_a_type, 0, grid->row);
    MPI_Type_free(&sub_a_type);
}

void distribute_matrix_b(const GridInfo* grid, Matrix* b, Matrix* sub_b) {
    int sub_cols = n3 / grid->dims[1];
    MPI_Datatype sub_b_type, sub_b_contig_type;
    
    MPI_Type_vector(n2, sub_cols, n3, MPI_DOUBLE, &sub_b_type);
    MPI_Type_commit(&sub_b_type);
    MPI_Type_contiguous(n2 * sub_cols, MPI_DOUBLE, &sub_b_contig_type);
    MPI_Type_commit(&sub_b_contig_type);

    if(grid->rank == 0) {
        for(int row = 0; row < n2; row++) {
            for(int col = 0; col < sub_cols; col++) {
                sub_b->data[row * sub_cols + col] = b->data[row * n3 + col];
            }
        }
        for(int i = 1; i < grid->dims[1]; i++) {
            MPI_Send(b->data + sub_cols * i, 1, sub_b_type, 
                    i, 0, grid->row);
        }
    }
    
    if(grid->coords[0] == 0 && grid->coords[1] != 0) {
        MPI_Recv(sub_b->data, 1, sub_b_contig_type, 
                0, 0, grid->row, MPI_STATUS_IGNORE);
    }
    
    MPI_Bcast(sub_b->data, 1, sub_b_contig_type, 0, grid->col);
    
    MPI_Type_free(&sub_b_type);
    MPI_Type_free(&sub_b_contig_type);
}

void local_matrix_multiply(const Matrix* a, const Matrix* b, Matrix* c) {
    for(int i = 0; i < a->rows; i++) {
        for(int j = 0; j < b->cols; j++) {
            c->data[i * c->cols + j] = 0;
            for(int k = 0; k < a->cols; k++) {
                c->data[i * c->cols + j] += 
                    a->data[i * a->cols + k] * 
                    b->data[k * b->cols + j];
            }
        }
    }
}

void gather_results(const GridInfo* grid, Matrix* sub_c, Matrix* c) {
    int sub_rows = n1 / grid->dims[0];
    int sub_cols = n3 / grid->dims[1];
    
    MPI_Datatype sub_c_type, sub_c_rows_type, sub_c_stride_type;
    MPI_Type_contiguous(sub_rows * sub_cols, MPI_DOUBLE, &sub_c_type);
    MPI_Type_commit(&sub_c_type);
    
    MPI_Type_contiguous(sub_rows * n3, MPI_DOUBLE, &sub_c_rows_type);
    MPI_Type_commit(&sub_c_rows_type);
    
    MPI_Type_vector(sub_rows, sub_cols, n3, MPI_DOUBLE, &sub_c_stride_type);
    MPI_Type_commit(&sub_c_stride_type);

    Matrix sub_c_rows = {0};
    if(grid->coords[1] == 0) {
        sub_c_rows.data = malloc(sub_rows * n3 * sizeof(double));
        sub_c_rows.rows = sub_rows;
        sub_c_rows.cols = n3;
        
        for(int i = 0; i < sub_rows; i++) {
            for(int j = 0; j < sub_cols; j++) {
                sub_c_rows.data[i * n3 + j] = sub_c->data[i * sub_cols + j];
            }
        }
        
        for(int i = 1; i < grid->dims[1]; i++) {
            MPI_Recv(sub_c_rows.data + sub_cols * i, 1, sub_c_stride_type,
                    i, 0, grid->row, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(sub_c->data, 1, sub_c_type, 0, 0, grid->row);
    }

    if(grid->coords[1] == 0) {
        MPI_Gather(sub_c_rows.data, 1, sub_c_rows_type,
                  c->data, 1, sub_c_rows_type,
                  0, grid->col);
    }

    free(sub_c_rows.data);
    MPI_Type_free(&sub_c_type);
    MPI_Type_free(&sub_c_rows_type);
    MPI_Type_free(&sub_c_stride_type);
}

int main(int argc, char** argv) {
    GridInfo grid = {0};
    setup_mpi_environment(&grid, argc, argv);
    validate_dimensions(&grid);

    Matrix a = {0}, b = {0}, c = {0};
    Matrix sub_a = {0}, sub_b = {0}, sub_c = {0};
    double start_time = 0;

    // Initialize matrices
    int sub_rows = n1 / grid.dims[0];
    int sub_cols = n3 / grid.dims[1];
    
    sub_a.rows = sub_rows;
    sub_a.cols = n2;
    sub_a.data = malloc(sub_rows * n2 * sizeof(double));
    
    sub_b.rows = n2;
    sub_b.cols = sub_cols;
    sub_b.data = malloc(n2 * sub_cols * sizeof(double));
    
    sub_c.rows = sub_rows;
    sub_c.cols = sub_cols;
    sub_c.data = calloc(sub_rows * sub_cols, sizeof(double));

    if(grid.rank == 0) {
        a.rows = n1;
        a.cols = n2;
        a.data = malloc(n1 * n2 * sizeof(double));
        for(int i = 0; i < n1 * n2; i++) a.data[i] = i;

        b.rows = n2;
        b.cols = n3;
        b.data = malloc(n2 * n3 * sizeof(double));
        for(int i = 0; i < n2 * n3; i++) b.data[i] = i;

        c.rows = n1;
        c.cols = n3;
        c.data = malloc(n1 * n3 * sizeof(double));

        start_time = MPI_Wtime();
    }

    // Data distribution
    distribute_matrix_a(&grid, &a, &sub_a);
    distribute_matrix_b(&grid, &b, &sub_b);

    // Local computation
    local_matrix_multiply(&sub_a, &sub_b, &sub_c);

    // Result collection
    if(grid.rank == 0) {
        gather_results(&grid, &sub_c, &c);
    } else {
        gather_results(&grid, &sub_c, &c);
    }

    // Cleanup and output
    if(grid.rank == 0) {
        double end_time = MPI_Wtime();
        printf("Execution time: %.4f seconds\n", end_time - start_time);
        // print_matrix(&c);
        free(a.data);
        free(b.data);
        free(c.data);
    }

    free(sub_a.data);
    free(sub_b.data);
    free(sub_c.data);

    MPI_Comm_free(&grid.grid);
    MPI_Comm_free(&grid.row);
    MPI_Comm_free(&grid.col);
    MPI_Finalize();
    return 0;
}
