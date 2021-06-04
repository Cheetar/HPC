#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <cassert>
#include <string.h>
#include <iostream>
#include <fstream>
#include "densematgen.h"
#include <vector>
#include <cstdlib>
#include "utils.h"

void multiplyColA(SparseMatrixFrag* A, DenseMatrixFrag* B, DenseMatrixFrag* C);

SparseMatrixFrag* shiftColA(SparseMatrixFrag* A, int myRank, int numProcesses, int round, int c);

DenseMatrixFrag* gatherResultColA(int myRank, int numProcesses, DenseMatrixFrag* C);

void colA(char* sparse_matrix_file, int seed, int c, int e, bool g, double g_val, bool verbose, int myRank, int numProcesses);