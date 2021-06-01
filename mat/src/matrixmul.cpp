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

const static bool DEBUG = true;
const static int ROOT_PROCESS = 0;


int getFirstColIdxIncl(int myRank, int numProcesses, int n) {
    return myRank * n/numProcesses;
}

int getLastColIdxExcl(int myRank, int numProcesses, int n) {
    return getFirstColIdxIncl(myRank + 1, numProcesses, n);
}


class SparseMatrixFrag{
    public:
        int n;
        int numElems;
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* values;
        int* colIdx;
        int* rowIdx;

        SparseMatrixFrag(int n, int numElems, double* values, int* rowIdx, int* colIdx, int firstColIdxIncl, int lastColIdxExcl) {
            this->n = n;
            this->numElems = numElems;
            this->values = values;
            this->colIdx = colIdx;
            this->rowIdx = rowIdx;
            this->firstColIdxIncl = firstColIdxIncl;
            this->lastColIdxExcl = lastColIdxExcl;
        }

        ~SparseMatrixFrag() {
            delete(this->values);
            delete(this->rowIdx);
            delete(this->colIdx);
        }

        std::vector<SparseMatrixFrag> chunk(int numChunks) {
            assert (this->n % numChunks == 0);
            std::vector<SparseMatrixFrag> chunks;
            for (int chunkId=0; chunkId<numChunks; chunkId++) {
                int firstColIdxIncl = getFirstColIdxIncl(chunkId, numChunks, this->n);
                int lastColIdxExcl = getLastColIdxExcl(chunkId, numChunks, this->n);
                std::vector<double> chunkValues;
                std::vector<int> chunkColIdx;
                std::vector<int> chunkRowIdx;
                int numElementsInChunk = 0;
                chunkRowIdx.push_back(0);
                for (int row=0; row<n; row++) {
                    int idx = this->rowIdx[row];
                    int nextIdx = this->rowIdx[row+1];
                    for (int i=idx; i<nextIdx; i++) {
                        if ((this->colIdx[i] >= firstColIdxIncl) && (this->colIdx[i] < lastColIdxExcl)) {
                            numElementsInChunk++;
                            chunkValues.push_back(this->values[i]);
                            chunkColIdx.push_back(this->colIdx[i]);
                        }
                    }
                    chunkRowIdx.push_back(numElementsInChunk);
                }

                double* values = new double[numElementsInChunk];
                int* rowIdx = new int[n + 1];
                int* colIdx = new int[numElementsInChunk];
                std::copy(chunkValues.begin(), chunkValues.end(), values);
                std::copy(chunkRowIdx.begin(), chunkRowIdx.end(), rowIdx);
                std::copy(chunkColIdx.begin(), chunkColIdx.end(), colIdx);
            
                SparseMatrixFrag chunk = SparseMatrixFrag(this->n, numElementsInChunk, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
                chunks.push_back(chunk);
            }
            return chunks;
        }

        void printout() {
            for (int i=0; i<this->numElems; i++)
                std::cout << this->values[i] << " ";
            std::cout << std::endl;

            for (int i=0; i<this->numElems; i++)
                std::cout << this->colIdx[i] << " ";
            std::cout << std::endl;

            for (int i=0; i<this->n+1; i++)
                std::cout << this->rowIdx[i] << " ";
            std::cout << std::endl;
        }
};


class DenseMatrixFrag{
    public:
        int n;  // Matrix of size n x n
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* data;  // Data aligned by columns i.e. first n entries represent first column

        DenseMatrixFrag(int n, int myRank, int numProcesses, int seed) {
            this->n = n;
            this->firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n);
            this->lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n);
            this->data = new double[n*n/numProcesses];
            for (int global_col=this->firstColIdxIncl; global_col<this->lastColIdxExcl; global_col++) {
                for (int row=0; row<n; row++) {
                    int local_col = global_col - this->firstColIdxIncl;
                    this->data[local_col*n + row] = generate_double(seed, row, global_col);
                }
            }
        }

        ~DenseMatrixFrag() {
            delete(this->data);
        }

        void printout() {
            for (int row=0; row<this->n; row++) {
                for (int col = this->firstColIdxIncl; col<lastColIdxExcl; col++) {
                    int local_col = col - this->firstColIdxIncl;
                    std::cout << data[local_col*n + row] << " ";
                }
                std::cout << std::endl;
            }
        }
};



void multiply(SparseMatrixFrag& A, DenseMatrixFrag& B, DenseMatrixFrag& C) {

}


int main(int argc, char * argv[]) {
    int numProcesses, myRank, seed, c, e, n;
    bool g = false;
    bool verbose = false;
    bool inner = false;
    char* sparse_matrix_file; 
    double g_val;

    MPI_Status *status;

    /* MPI initialization */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    /* Read program arguments */
    // Format: matrixmul -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e exponent [-g ge_value] [-v] [-i]
    assert((9 <= argc) && (argc <= 13));

    assert(strcmp(argv[1], "-f") == 0);
    sparse_matrix_file = argv[2];

    assert(strcmp(argv[3], "-s") == 0);
    seed = atoi(argv[4]);

    assert(strcmp(argv[5], "-c") == 0);
    c = atoi(argv[6]);

    assert(strcmp(argv[7], "-e") == 0);
    e = atoi(argv[8]);

    if ((argc >= 11) && (strcmp(argv[9], "-g") == 0)) {
        g = true;
        g_val = atof(argv[10]);
    }

    for (int i=9; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-i") == 0) {
            inner = true;
        }
    }
    
    assert (!(g && verbose));  // g and verbose parameters are exclusive

    if (DEBUG) {
        std::cout << "Argc: " << argc << std::endl
                  << "sparse_matrix_file: " << sparse_matrix_file << std::endl
                  << "seed: " << seed << std::endl
                  << "c: " << c << std::endl
                  << "e: " << e << std::endl
                  << "g: " << g << std::endl
                  << "g_val: " << g_val << std::endl
                  << "v: " << verbose << std::endl
                  << "i: " << inner << std::endl;
    }

    // Root process reads and distributes sparse matrix A
    if (myRank == ROOT_PROCESS) {
        // Read matrix from file
        int elems, d;
        std::ifstream ReadFile(sparse_matrix_file);

        ReadFile >> n;
        ReadFile >> n;  // A is a square matrix
        ReadFile >> elems;
        ReadFile >> d;

        double* values = new double[elems];
        int* rowIdx = new int[n + 1];
        int* colIdx = new int[elems];
        
        for (int i=0; i<elems; i++)
            ReadFile >> values[i];

        for (int i=0; i<n+1; i++)
            ReadFile >> rowIdx[i];

        for (int i=0; i<elems; i++)
            ReadFile >> colIdx[i];
        
        ReadFile.close();

        SparseMatrixFrag A = SparseMatrixFrag(n, elems, values, rowIdx, colIdx, 0, n);

        if (DEBUG)
            A.printout(); 
    }
    
    // Broadcast matrix size
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );
    
    // TODO Distribute chunks over all processes
    if (myRank == ROOT_PROCESS) {
        std::vector<SparseMatrixFrag> chunks = A.chunk(numProcesses);
        for(std::vector<SparseMatrixFrag>::iterator chunk = chunks.begin(); chunk != chunks.end(); ++chunk) {
            it.printout();
        } 
    } else {

    }

    // TODO for simpicity, later pad with zeros
    assert(n % numProcesses == 0);

    // Generate fragment of dense matrix
    DenseMatrixFrag B = DenseMatrixFrag(n, myRank, numProcesses, seed);
    /*if (DEBUG)
        B.printout();*/

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
