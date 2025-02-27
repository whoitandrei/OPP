#!/bin/bash

SRC_DIR="src"
BIN_DIR="bin"
mkdir -p $BIN_DIR

g++ -Wall -Wextra -O3 -o $BIN_DIR/SLAU $SRC_DIR/SLAU.cpp
g++ -Wall -Wextra -O3 -fopenmp -o $BIN_DIR/omp1 $SRC_DIR/omp1.cpp
g++ -Wall -Wextra -O3 -fopenmp -o $BIN_DIR/omp2 $SRC_DIR/omp2.cpp

if [ $? -eq 0 ]; then
	echo "file compiled in [$BIN_DIR] directory"
else
	echo "error"
