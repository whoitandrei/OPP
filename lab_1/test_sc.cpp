    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <ctime>
    #include <omp.h>
    #include <numeric>  

    #define EPS 0.00001
    #define TAU (1.9 / (size + 1))

    #ifndef SCHEDULE_TYPE
        #define SCHEDULE_TYPE static
    #endif

    #ifndef CHUNK_SIZE
        #define CHUNK_SIZE 10
    #endif

    std::vector<double> mulMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec, int size) {
        std::vector<double> result(size, 0.0);

        #pragma omp parallel for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }

    std::vector<double> subVector(const std::vector<double>& vec1, const std::vector<double>& vec2, int size) {
        std::vector<double> result(size);

        #pragma omp parallel for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
        for (int i = 0; i < size; ++i) {
            result[i] = vec1[i] - vec2[i];
        }
        return result;
    }

    void mulConstVector(std::vector<double>& vec, double num, int size) {
        #pragma omp parallel for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
        for (int i = 0; i < size; ++i) {
            vec[i] *= num;
        }
    }

    void initMatrix(std::vector<std::vector<double>>& matrix, int size) {
        #pragma omp parallel for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = (i == j) ? 2 : 1;
            }
        }
    }

    void initVector(std::vector<double>& vec, int size, double value) {
        #pragma omp parallel for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
        for (int i = 0; i < size; ++i) {
            vec[i] = value;
        }
    }

    void findGoodSolve(const std::vector<std::vector<double>>& A, std::vector<double>& X, const std::vector<double>& B, int size) {
        std::vector<double> AX_B = subVector(mulMatrixVector(A, X, size), B, size);

        while (sqrt(std::inner_product(AX_B.begin(), AX_B.end(), AX_B.begin(), 0.0)) / sqrt(std::inner_product(B.begin(), B.end(), B.begin(), 0.0)) >= EPS) {
            mulConstVector(AX_B, TAU, size);
            X = subVector(X, AX_B, size);
            AX_B = subVector(mulMatrixVector(A, X, size), B, size);
        }
    }

    int main() {
        int size = 1000;  
        int T = 4;        

        omp_set_num_threads(T);

        std::vector<std::vector<double>> A(size, std::vector<double>(size));
        std::vector<double> X(size);
        std::vector<double> B(size);

        initMatrix(A, size);
        initVector(B, size, size + 1);
        initVector(X, size, 0.0);

        double start = omp_get_wtime();
        findGoodSolve(A, X, B, size);
        double end = omp_get_wtime();

        std::cout << "\nTime: " << end - start << " seconds" << std::endl;
        std::cout << "Answer (some nums from vector X):" << std::endl;
        std::cout << X[0] << " " << X[size/3] << " " << X[size/2] << " " << X[size-1] << std::endl;

        return 0;
    }
