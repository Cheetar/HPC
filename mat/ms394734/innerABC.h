#ifndef __MIMUW_INNERABC_H__
#define __MIMUW_INNERABC_H__

void multiplyColA(SparseMatrixFragByRow* A, DenseMatrixFrag* B, DenseMatrixFrag* C);

SparseMatrixFragByRow* shiftInnerABC(SparseMatrixFragByRow* A, int* cache, int myRank, int numProcesses, int round, int c);

DenseMatrixFrag* gatherResultInnerABC(int myRank, int numProcesses, DenseMatrixFrag* C);

void innerABC(char* sparse_matrix_file, int seed, int c, int e, bool g, double g_val, bool verbose, int myRank, int numProcesses);

#endif