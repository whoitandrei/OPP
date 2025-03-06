#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void multiplyMatrixAndVector(double* A_local, int myLocalN, double* x_local, double* Ax_local, int* counts, int* displs, int numberOfProcesses, int rank, int N, MPI_Comm comm) {
    for (int i = 0; i < myLocalN; i++) {
        Ax_local[i] = 0.0;
    }

    int maxLocalN = ((N % numberOfProcesses) == 0) ? (N / numberOfProcesses) : (N / numberOfProcesses + 1);
    double* sendBuf = (double*)malloc(maxLocalN * sizeof(double));
    double* recvBuf = (double*)malloc(maxLocalN * sizeof(double));
    int currentSize = counts[rank];
    memcpy(sendBuf, x_local, currentSize * sizeof(double));
    int currentOwner = rank;

    for (int step = 0; step < numberOfProcesses; step++) {
        int blockSize = counts[currentOwner];
        int blockDisp = displs[currentOwner];
        for (int i = 0; i < myLocalN; i++) {
            double part = 0.0;
            for (int k = 0; k < blockSize; k++) {
                int globalIndex = blockDisp + k;
                part += A_local[i * N + globalIndex] * sendBuf[k];
            }
            Ax_local[i] += part;
        }
        int leftNeighbour = (rank - 1 + numberOfProcesses) % numberOfProcesses;
        int rightNeighbour = (rank + 1) % numberOfProcesses;
        int sendCurSizeAndCurOwner[2] = {currentSize, currentOwner};
        int recvPrevSizeAndPrevOwner[2];
        MPI_Sendrecv(sendCurSizeAndCurOwner, 2, MPI_INT, rightNeighbour, 0, recvPrevSizeAndPrevOwner, 2, MPI_INT, leftNeighbour, 0, comm, MPI_STATUS_IGNORE);
        int nextSize = recvPrevSizeAndPrevOwner[0];
        int nextOwner = recvPrevSizeAndPrevOwner[1];
        MPI_Sendrecv(sendBuf, currentSize, MPI_DOUBLE, rightNeighbour, 1, recvBuf, nextSize, MPI_DOUBLE, leftNeighbour, 1, comm, MPI_STATUS_IGNORE);
        currentSize = nextSize;
        currentOwner = nextOwner;
        memcpy(sendBuf, recvBuf, currentSize * sizeof(double));
    }
    free(sendBuf);
    free(recvBuf);
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
    double* A = NULL;
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
    }
    double* x_local = (double*)malloc(countLinesForProc[rank] * sizeof(double));
    double* b_local = (double*)malloc(countLinesForProc[rank] * sizeof(double));

    double g_x = 0.0;
    double abs_b = 0.0;
    if (rank == 0) {
        double* full_b = (double*)malloc(N * sizeof(double));
        double* full_x = (double*)malloc(N * sizeof(double));
        firstInit(A, full_b, full_x, N);
        MPI_Scatterv(full_b, countLinesForProc, offsetArray, MPI_DOUBLE, b_local, countLinesForProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(full_x, countLinesForProc, offsetArray, MPI_DOUBLE, x_local, countLinesForProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(full_b);
        free(full_x);
        g_x = 1;
    } else {
        MPI_Scatterv(NULL, countLinesForProc, offsetArray, MPI_DOUBLE, b_local, countLinesForProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, countLinesForProc, offsetArray, MPI_DOUBLE, x_local, countLinesForProc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double localAbsSqr = countAbsSqr(b_local, countLinesForProc[rank]);
    double globalAbsSqr = 0.0;
    MPI_Allreduce(&localAbsSqr, &globalAbsSqr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    abs_b = sqrt(globalAbsSqr);

    MPI_Bcast(&g_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A, countOfMatrElem_s, countOfSkipedElem, MPI_DOUBLE, partOfA, countOfMatrElem_s[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* Ax_b = (double*)malloc(countLinesForProc[rank] * sizeof(double));
    double* partOfX = (double*)malloc(countLinesForProc[rank] * sizeof(double));

    double absOfSum = 0;
    int iterationCount = 0;

    while (g_x > eps && iterationCount < maxIterationsCount) {
        multiplyMatrixAndVector(partOfA, countLinesForProc[rank], x_local, Ax_b, countLinesForProc, offsetArray, size, rank, N, MPI_COMM_WORLD);

        for (int i = 0; i < countLinesForProc[rank]; ++i) {
            Ax_b[i] = Ax_b[i] - b_local[i];
            partOfX[i] = x_local[i] - taul * Ax_b[i];
        }

        memcpy(x_local, partOfX, countLinesForProc[rank] * sizeof(double));

        double localAbs = countAbsSqr(Ax_b, countLinesForProc[rank]);
        MPI_Allreduce(&localAbs, &absOfSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        g_x = sqrt(absOfSum) / abs_b;

        if (rank == 0) {
            iterationCount++;
        }

        MPI_Bcast(&iterationCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (iterationCount < maxIterationsCount) {
            double endTime = MPI_Wtime();
            printf("time: %f sec\n", endTime - startTime);
        } else {
            printf("No converge, change taul sign\n");
        }

        double* x_global = (double*)malloc(N * sizeof(double));
        MPI_Gatherv(x_local, countLinesForProc[rank], MPI_DOUBLE, x_global, countLinesForProc, offsetArray, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double maxDifference = 0.0;
        for (int i = 0; i < N; ++i) {
            double difference = fabs(x_global[i] - 1.0);
            if (difference > maxDifference) {
                maxDifference = difference;
            }
        }
        printf("max difference: %lf\n", maxDifference);

        free(x_global);
    }

    free(countLinesForProc);
    free(offsetArray);
    free(countOfMatrElem_s);
    free(countOfSkipedElem);
    free(partOfA);
    if (rank == 0) {
        free(A);
    }
    free(x_local);
    free(b_local);
    free(Ax_b);
    free(partOfX);

    MPI_Finalize();

    return 0;
}