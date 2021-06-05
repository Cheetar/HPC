#ifndef __MIMUW_COLA_H__
#define __MIMUW_COLA_H__

void multiplyColA(SparseMatrixFrag* A, DenseMatrixFrag* B, DenseMatrixFrag* C);

SparseMatrixFrag* shiftColA(SparseMatrixFrag* A, int* cache, int myRank, int numProcesses, int round, int c);

DenseMatrixFrag* gatherResultColA(int myRank, int numProcesses, DenseMatrixFrag* C);

void colA(char* sparse_matrix_file, int seed, int c, int e, bool g, double g_val, bool verbose, int myRank, int numProcesses);

#endif