#ifndef __MIMUW_UTILS_H__
#define __MIMUW_UTILS_H__

int calcFirstColIdxIncl(int myRank, int numProcesses, int n);
int getFirstColIdxIncl(int myRank, int numProcesses, int n, int round=0, int c=1);
int getLastColIdxExcl(int myRank, int numProcesses, int n, int round=0, int c=1);

int getChunkSize(int *cache, int chunkNum);
int getChunkNumber(int myRank, int numProcesses, int round=0, int c=1);

class SparseMatrixFrag {
    public:
        int n;
        int pad_size;
        int numElems;
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* values;
        int* rowIdx;
        int* colIdx;

        SparseMatrixFrag(int n, int pad_size, int numElems, double* values, int* rowIdx, int* colIdx, int firstColIdxIncl, int lastColIdxExcl);

        SparseMatrixFrag(int n, int pad_size, int firstColIdxIncl, int lastColIdxExcl);

        ~SparseMatrixFrag();

        std::vector<SparseMatrixFrag*> chunk(int numChunks);

        void printout();
};

class SparseMatrixFragByRow {
    public:
        int n;
        int pad_size;
        int numElems;
        int numRows;
        int firstRowIdxIncl;
        int lastRowIdxExcl;
        double* values;
        int* rowIdx;
        int* colIdx;

        SparseMatrixFragByRow(int n, int pad_size, int numElems, double* values, int* rowIdx, int* colIdx, int firstRowIdxIncl, int lastRowIdxExcl);

        SparseMatrixFragByRow(int n, int pad_size, int firstColIdxIncl, int lastColIdxExcl);

        ~SparseMatrixFragByRow();

        std::vector<SparseMatrixFragByRow*> chunk(int numChunks);

        void printout();
};

class DenseMatrixFrag{
    public:
        int n;  // Matrix of size n x n
        int pad_size;
        int numElems;
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* data;  // Data aligned by columns i.e. first n entries represent first column

        DenseMatrixFrag(int n, int pad_size, int firstColIdxIncl, int lastColIdxExcl, int seed=0, bool initialize=true);

        ~DenseMatrixFrag();

        void add(int row, int col, double val);

        double get(int row, int col);

        void addChunk(DenseMatrixFrag* chunk);

        void printout();

        // Prints number of elements greater or equal th
        void printout(double th);
};

#endif
