#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

#define EPS 0.00001
#define TAU (1.9 / (size + 1))

/* math with vectors funcs */
std::vector<double> mulMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec, int size) {
    std::vector<double> result(size, 0.0);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}
    
std::vector<double> subVector(const std::vector<double>& vec1, const std::vector<double>& vec2, int size) {
    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

void mulConstVector(std::vector<double>& vec, double num, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] *= num;
    }
}

/* funcs with matrices (init etc.) */
void initMatrix(std::vector<std::vector<double>>* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) (*matrix)[i][j] = 2;
            else (*matrix)[i][j] = 1;
        }
    }
}

void initVectorB(std::vector<double>* B, int size) {
    for (int i = 0; i < size; ++i) {
        (*B)[i] = size + 1;
    }
}

void initVectorX(std::vector<double>* X, int size) {
    for (int i = 0; i < size; ++i) {
        (*X)[i] = 0.;
    }
}

/* funcs for solve approximating to EPS */
double normOfVector(const std::vector<double>& vec, int size) {
    double result = 0;
    for (int i = 0; i < size; ++i) {
        result += (vec[i] * vec[i]);
    }
    return sqrt(result);
}

bool isGoodSolve(const std::vector<double>& AX_B, const std::vector<double>& B, int size) {
    return (normOfVector(AX_B, size) / normOfVector(B, size)) < EPS ? true : false;
}

// (main loop of program)
void findGoodSolve(const std::vector<std::vector<double>>& A, std::vector<double>* X, const std::vector<double>& B, int size) {
    std::vector<double> AX_B = subVector(mulMatrixVector(A, *X, size), B, size);

    while (!isGoodSolve(AX_B, B, size)) {
        mulConstVector(AX_B, TAU, size);
        *X = subVector(*X, AX_B, size);
        AX_B = subVector(mulMatrixVector(A, *X, size), B, size);
    }
}
    
/* main func */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    int size = std::stoi(argv[1]);

    std::vector<std::vector<double>> A(size, std::vector<double>(size));
    std::vector<double> X(size);
    std::vector<double> B(size);

    initMatrix(&A, size);
    initVectorB(&B, size);
    initVectorX(&X, size);

    findGoodSolve(A, &X, B, size);

    double maxDifference = 0.;
    for (int i = 0; i < size; i++){
        double difference = fabs(B[i] - X[i]);
        if (difference > maxDifference)
            maxDifference = difference;
    }
    std::cout << "max difference: " << maxDifference << std::endl;  

    return 0;
}