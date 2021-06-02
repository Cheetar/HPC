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
const static int TAG = 13;


int getFirstColIdxIncl(int myRank, int numProcesses, int n) {
    return myRank * n/numProcesses;
}

int getFirstColIdxIncl(int myRank, int numProcesses, int n, int round) {
    return getFirstColIdxIncl((myRank + round) % numProcesses, numProcesses, n);
}

int getLastColIdxExcl(int myRank, int numProcesses, int n) {
    return getFirstColIdxIncl(myRank + 1, numProcesses, n);
}

int getLastColIdxExcl(int myRank, int numProcesses, int n, int round) {
    return getFirstColIdxIncl(((myRank + round) % numProcesses) + 1, numProcesses, n);
}


class SparseMatrixFrag{
    public:
        int n;
        int numElems;
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* values;
        int* rowIdx;
        int* colIdx;

        SparseMatrixFrag(int n, int numElems, double* values, int* rowIdx, int* colIdx, int firstColIdxIncl, int lastColIdxExcl) {
            this->n = n;
            this->numElems = numElems;
            this->values = values;
            this->rowIdx = rowIdx;
            this->colIdx = colIdx;
            this->firstColIdxIncl = firstColIdxIncl;
            this->lastColIdxExcl = lastColIdxExcl;
        }

        ~SparseMatrixFrag() {
            free(this->values);
            free(this->rowIdx);
            free(this->colIdx);
        }

        std::vector<SparseMatrixFrag*> chunk(int numChunks) {
            assert (this->n % numChunks == 0);
            std::vector<SparseMatrixFrag*> chunks;
            for (int chunkId=0; chunkId<numChunks; chunkId++) {
                int firstColIdxIncl = getFirstColIdxIncl(chunkId, numChunks, this->n);
                int lastColIdxExcl = getLastColIdxExcl(chunkId, numChunks, this->n);
                std::vector<double> chunkValues;
                std::vector<int> chunkRowIdx;
                std::vector<int> chunkColIdx;
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

                double* values = (double*)malloc(sizeof(double) * numElementsInChunk);
                int* rowIdx = (int*)malloc(sizeof(int) * (n+1));
                int* colIdx = (int*)malloc(sizeof(int) * numElementsInChunk);
                std::copy(chunkValues.begin(), chunkValues.end(), values);
                std::copy(chunkRowIdx.begin(), chunkRowIdx.end(), rowIdx);
                std::copy(chunkColIdx.begin(), chunkColIdx.end(), colIdx);

                SparseMatrixFrag *chunk = new SparseMatrixFrag(this->n, numElementsInChunk, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
                chunks.push_back(chunk);
            }
            return chunks;
        }

        void printout() {
            for (int i=0; i<this->numElems; i++)
                std::cout << this->values[i] << " ";
            std::cout << std::endl;

            for (int i=0; i<this->n+1; i++)
                std::cout << this->rowIdx[i] << " ";
            std::cout << std::endl;

            for (int i=0; i<this->numElems; i++)
                std::cout << this->colIdx[i] << " ";
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

        void add(int row, int col, double val) {
            assert(col >= this->firstColIdxIncl && col < this->lastColIdxExcl);
            int local_col = col - this->firstColIdxIncl;
            this->data[local_col*this->n + row] += val;
        }

        double get(int row, int col) {
            assert(col >= this->firstColIdxIncl && col < this->lastColIdxExcl);
            int local_col = col - this->firstColIdxIncl;
            return this->data[local_col*this->n + row];
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

void multiplyColA(SparseMatrixFrag* A, DenseMatrixFrag* B, DenseMatrixFrag* C) {
    for (int row=B->firstColIdxIncl; row<B->lastColIdxExcl; row++) {
        int curRow = A->rowIdx[row];
        int nextRow = A->rowIdx[row + 1];
        for (int col=B->firstColIdxIncl; col<B->lastColIdxExcl; col++) {
            double val = 0;
            for (int j=curRow; j<nextRow; j++) {
                int elemCol = A->colIdx[j];

                double A_val = A->values[j];
                double B_val = B->get(elemCol, row);
                val += A_val * B_val;
            }
            C->add(row, col, val);
        }
    }
}

void shiftColA(SparseMatrixFrag* A, int myRank, int numProcesses, int round) {
    
    int prevProcessNo = myRank > 0 ? myRank - 1 : numProcesses - 1;
    int nextProcessNo = myRank < numProcesses - 1 ? myRank + 1 : 0;
    int n = A->n;

    MPI_Request requests[8];
    MPI_Status statuses[8];

    MPI_Isend(
        &A->numElems,
        1,
        MPI_INT,
        nextProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[0]
    );
    MPI_Isend(
        A->values,
        A->numElems,
        MPI_DOUBLE,
        nextProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[1]
    );
    MPI_Isend(
        A->rowIdx,
        n + 1,
        MPI_INT,
        nextProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[2]
    );
    MPI_Isend(
        A->colIdx,
        A->numElems,
        MPI_INT,
        nextProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[3]
    );


    // Receive
    int chunkNumElems;
    MPI_Irecv(
        &chunkNumElems,
        1,
        MPI_INT,
        prevProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[4]
    );
    MPI_Wait(&requests[4], &statuses[4]);

    double* values = new double[chunkNumElems];
    int* rowIdx = new int[A->n + 1];
    int* colIdx = new int[chunkNumElems];

    MPI_Irecv(
        values,
        chunkNumElems,
        MPI_DOUBLE,
        prevProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[5]
    );
    MPI_Irecv(
        rowIdx,
        n + 1,
        MPI_INT,
        prevProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[6]
    );
    MPI_Irecv(
        colIdx,
        chunkNumElems,
        MPI_INT,
        prevProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requests[7]
    );
    MPI_Waitall(8, requests, statuses);

    // Replace chunk of data
    delete A;
    int firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n, round);
    int lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n, round);
    A = new SparseMatrixFrag(n, chunkNumElems, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
}

int main(int argc, char * argv[]) {
    int numProcesses, myRank, seed, c, e, n, chunkNumElems;
    bool g = false;
    bool verbose = false;
    bool inner = false;
    char* sparse_matrix_file; 
    double g_val;

    MPI_Status status;
    SparseMatrixFrag* whole_A;
    SparseMatrixFrag* A;

    /* MPI initialization */
    MPI_Init(&argc, &argv);
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

        whole_A = new SparseMatrixFrag(n, elems, values, rowIdx, colIdx, 0, n);
        if (DEBUG)
            whole_A->printout();   
    } 

    // Broadcast matrix size
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    // Distribute chunks of A over all processes
    if (myRank == ROOT_PROCESS) {
        std::vector<SparseMatrixFrag*> chunks = whole_A->chunk(numProcesses);
        for (int processNum=1; processNum<numProcesses; processNum++){
            SparseMatrixFrag* chunk = chunks[processNum];
            chunkNumElems = chunk->numElems;

            // Send number of elements in a chunk 
            MPI_Send(
                &chunkNumElems,
                1,
                MPI_INT,
                processNum,
                TAG,
                MPI_COMM_WORLD
            );
            MPI_Send(
                chunk->values,
                chunkNumElems,
                MPI_DOUBLE,
                processNum,
                TAG,
                MPI_COMM_WORLD
            );
            MPI_Send(
                chunk->rowIdx,
                n+1,
                MPI_INT,
                processNum,
                TAG,
                MPI_COMM_WORLD
            );
            MPI_Send(
                chunk->colIdx,
                chunkNumElems,
                MPI_INT,
                processNum,
                TAG,
                MPI_COMM_WORLD
            );
        }

        // Initialize chunk of ROOT process
        A = chunks[0];
    } else {
        MPI_Recv(
            &chunkNumElems,
            1,
            MPI_INT,
            ROOT_PROCESS,
            TAG,
            MPI_COMM_WORLD,
            &status
        );

        double* values = new double[chunkNumElems];
        int* rowIdx = new int[n + 1];
        int* colIdx = new int[chunkNumElems];

        MPI_Recv(
            values,
            chunkNumElems,
            MPI_DOUBLE,
            ROOT_PROCESS,
            TAG,
            MPI_COMM_WORLD,
            &status
        );
        MPI_Recv(
            rowIdx,
            n+1,
            MPI_INT,
            ROOT_PROCESS,
            TAG,
            MPI_COMM_WORLD,
            &status
        );
        MPI_Recv(
            colIdx,
            chunkNumElems,
            MPI_INT,
            ROOT_PROCESS,
            TAG,
            MPI_COMM_WORLD,
            &status
        ); 

        int firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n);
        int lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n);
        A = new SparseMatrixFrag(n, chunkNumElems, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
    }

    std::cout << "myRank: " << myRank << std::endl;
    A->printout();

    // TODO for simpicity, later pad with zeros
    assert(n % numProcesses == 0);

    // Generate fragment of dense matrix
    DenseMatrixFrag* B = new DenseMatrixFrag(n, myRank, numProcesses, seed);
    /*if (DEBUG)
        B.printout();*/

    // ColA algorithm
    DenseMatrixFrag* C = new DenseMatrixFrag(n, myRank, numProcesses, 0);  // seed is 0, so matrix is all zeros
    for (int round=1; round<=numProcesses; round++) {
        multiplyColA(A, B, C);
        shiftColA(A, myRank, numProcesses, round);
    }
    C->printout();
    // DenseMatrixFrag* whole_C  = gatherResult(C);
    // whole_C->printout();

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
