#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>

#define EPS 0.00001
#define TAU (1.9 / (size + 1))

/* math with vectors funcs */
std::vector<double> mulMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec, int size) {
    std::vector<double> result(size, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<double> subVector(const std::vector<double>& vec1, const std::vector<double>& vec2, int size) {
    std::vector<double> result(size);

    #pragma omp parallel for 
    for (int i = 0; i < size; ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

void mulConstVector(std::vector<double>& vec, double num, int size) {
    #pragma omp parallel for 
    for (int i = 0; i < size; ++i) {
        vec[i] *= num;
    }
}

/* funcs with matrices (init etc.) */
void initMatrix(std::vector<std::vector<double>>* matrix, int size) {
    #pragma omp parallel for 
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) (*matrix)[i][j] = 2;
            else (*matrix)[i][j] = 1;
        }
    }
}

void initVectorB(std::vector<double>* B, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        (*B)[i] = size + 1;
    }
}

void initVectorX(std::vector<double>* X, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        (*X)[i] = 0.;
    }
}

/* funcs for solve approximating to EPS */
double normOfVector(const std::vector<double>& vec, int size) {
    double result = 0;
    #pragma omp parallel for reduction(+:result) 
    for (int i = 0; i < size; ++i) {
        result += (vec[i] * vec[i]);
    }
    return sqrt(result);
}

bool isGoodSolve(const std::vector<double>& AX_B, const std::vector<double>& B, int size) {
    return (normOfVector(AX_B, size) / normOfVector(B, size)) < EPS;
}

// (main loop of program)
void findGoodSolve(const std::vector<std::vector<double>>& A, std::vector<double>& X, const std::vector<double>& B, int size) {
        std::vector<double> AX_B(size);
        double norm_AB , norm_B;

        while (true) {
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += A[i][j] * X[j];
                }
                AX_B[i] = sum - B[i];
            }

            norm_AB = 0.0;
            norm_B = 0.0;


            #pragma omp parallel for reduction(+:norm_AB)
            for (int i = 0; i < size; ++i) {
                norm_AB += AX_B[i] * AX_B[i];
            }

            #pragma omp parallel for reduction(+:norm_B)
            for (int i = 0; i < size; ++i) {
                norm_B += B[i] * B[i];
            }

            norm_AB = sqrt(norm_AB);
            norm_B = sqrt(norm_B);

            if (norm_AB / norm_B < EPS) break;

            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                X[i] -= TAU * AX_B[i];
            }
        }
    
}


/* main func */
int main(int argc, char** argv) {
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N T\n";
        return 1;
    }

    int size = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);

    omp_set_num_threads(T);

    std::vector<std::vector<double>> A(size, std::vector<double>(size));
    std::vector<double> X(size);
    std::vector<double> B(size);

    initMatrix(&A, size);
    initVectorB(&B, size);
    initVectorX(&X, size);

    double start = omp_get_wtime();
    findGoodSolve(A, X, B, size);
    double end = omp_get_wtime();

    std::cout << "\nTime: " << end - start << " seconds" << std::endl << std::endl;
   
    
    double maxDifference = 0;
    for (int i = 0; i < size; i++){
        double difference = fabs(B[i] - X[i]);
        if (difference > maxDifference)
            maxDifference = difference;
    }
    std::cout << "max difference: " << maxDifference << std::endl;

    std::cout << X[0] << " " << X[size/3] << " " << X[size/2] << " " << X[size-1] << std::endl;

    return 0;
}
