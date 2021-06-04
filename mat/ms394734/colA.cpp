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

const static int ROOT_PROCESS = 0;
const static int TAG = 13;


void multiplyColA(SparseMatrixFrag* A, DenseMatrixFrag* B, DenseMatrixFrag* C) {
    if (A->numElems > 0) {
        for (int row=0; row<A->n; row++) {
            int curRow = A->rowIdx[row];
            int nextRow = A->rowIdx[row + 1];
            for (int col=B->firstColIdxIncl; col<B->lastColIdxExcl; col++) {
                double val = 0;
                for (int j=curRow; j<nextRow; j++) {
                    int elemCol = A->colIdx[j];

                    double A_val = A->values[j];
                    double B_val = B->get(elemCol, col);
                    val += A_val * B_val;
                }
                C->add(row, col, val);
            }
        }
    }
}

SparseMatrixFrag* shiftColA(SparseMatrixFrag* A, int myRank, int numProcesses, int round, int c) {
    int n = A->n;
    int pad_size = A->pad_size;
    int chunkNumElems;
    int firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n, round, c);
    int lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n, round, c);
    int groupSize = numProcesses/c;
    int groupRank = myRank % groupSize;
    int groupNumber = myRank / groupSize;
    int prevProcessNo = groupRank > 0 ? myRank - 1 : myRank + groupSize - 1;
    int nextProcessNo = groupRank < groupSize - 1 ? myRank + 1 : groupNumber * groupSize;
    double* values;
    int* rowIdx;
    int* colIdx;

    MPI_Request requestsChunkSize[2];
    MPI_Status statusesChunkSize[2];

    MPI_Request requestsRecv[3];
    MPI_Status statusesRecv[3];

    MPI_Request requestsSend[3];
    MPI_Status statusesSend[3];

    assert(A->numElems >= 0);

    // Share chunk size with next neighbour
    MPI_Isend(
        &A->numElems,
        1,
        MPI_INT,
        nextProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requestsChunkSize[0]
    );
    MPI_Irecv(
        &chunkNumElems,
        1,
        MPI_INT,
        prevProcessNo,
        TAG,
        MPI_COMM_WORLD,
        &requestsChunkSize[1]
    );
    MPI_Waitall(2, requestsChunkSize, statusesChunkSize);

    // Receive chunk
    if (chunkNumElems > 0) {
        values = new double[chunkNumElems];
        rowIdx = new int[n + 1];
        colIdx = new int[chunkNumElems];

        MPI_Irecv(
            values,
            chunkNumElems,
            MPI_DOUBLE,
            prevProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsRecv[0]
        );
        MPI_Irecv(
            rowIdx,
            n + 1,
            MPI_INT,
            prevProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsRecv[1]
        );
        MPI_Irecv(
            colIdx,
            chunkNumElems,
            MPI_INT,
            prevProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsRecv[2]
        );
    }

    // Send chunk
    if (A->numElems > 0) {
        MPI_Isend(
            A->values,
            A->numElems,
            MPI_DOUBLE,
            nextProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsSend[0]
        );
        MPI_Isend(
            A->rowIdx,
            n + 1,
            MPI_INT,
            nextProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsSend[1]
        );
        MPI_Isend(
            A->colIdx,
            A->numElems,
            MPI_INT,
            nextProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestsSend[2]
        );
    }

    if (chunkNumElems > 0)
        MPI_Waitall(3, requestsRecv, statusesRecv);
    if (A->numElems > 0)
        MPI_Waitall(3, requestsSend, statusesSend);

    delete(A);
    if (chunkNumElems > 0)
        A = new SparseMatrixFrag(n, pad_size, chunkNumElems, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
    else 
        A = new SparseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl);
    return A;
}

DenseMatrixFrag* gatherResultColA(int myRank, int numProcesses, DenseMatrixFrag* C) {
    int firstColIdxIncl, lastColIdxExcl;
    int n = C->n;
    if (myRank == ROOT_PROCESS) {
        MPI_Request requests[numProcesses - 1];
        MPI_Status statuses[numProcesses - 1];

        std::vector<DenseMatrixFrag*> chunks;
        chunks.push_back(C);  // Add chunk of ROOT process

        for (int processNum=1; processNum<numProcesses; processNum++) {
            firstColIdxIncl = getFirstColIdxIncl(processNum, numProcesses, n);
            lastColIdxExcl = getLastColIdxExcl(processNum, numProcesses, n);
            // Create empty placeholders for partial results
            DenseMatrixFrag* chunk = new DenseMatrixFrag(C->n, C->pad_size, firstColIdxIncl, lastColIdxExcl);
            chunks.push_back(chunk);
            MPI_Irecv(
                chunks[processNum]->data,
                chunks[processNum]->numElems,
                MPI_DOUBLE,
                processNum,
                TAG,
                MPI_COMM_WORLD,
                &requests[processNum - 1]
            );
        }
        MPI_Waitall(numProcesses - 1, requests, statuses);

        // seed is 0, so matrix is empty (filled with zeros)
        DenseMatrixFrag* whole_C = new DenseMatrixFrag(C->n, C->pad_size, 0 /*firstColIdxIncl*/, C->n /*lastColIdxExcl*/); 
        // Marge chunks into one final matrix
        for (int i=0; i<numProcesses; i++)
            whole_C->addChunk(chunks[i]);
        return whole_C;
    } else {
        MPI_Request request;
        assert(C->numElems > 0);

        MPI_Isend(
            C->data,
            C->numElems,
            MPI_DOUBLE,
            ROOT_PROCESS,
            TAG,
            MPI_COMM_WORLD,
            &request
        );

        return nullptr;
    }
}

void colA(char* sparse_matrix_file, int seed, int c, int e, bool g, double g_val, bool verbose, int myRank, int numProcesses) {
    int pad_size, chunkNumElems, org_n, n, firstColIdxIncl, lastColIdxExcl;

    MPI_Status status;
    SparseMatrixFrag* A;
    SparseMatrixFrag* whole_A;
    DenseMatrixFrag* B;
    DenseMatrixFrag* C;
    DenseMatrixFrag* whole_C;

    // Root process reads sparse matrix A
    if (myRank == ROOT_PROCESS) {
        // Read matrix from file
        int elems, d;
        std::ifstream ReadFile(sparse_matrix_file);

        ReadFile >> org_n;
        ReadFile >> org_n;  // A is a square matrix
        ReadFile >> elems;
        ReadFile >> d;

        // Pad matrix with zeroes
        pad_size = (org_n % numProcesses == 0) ? 0 : numProcesses - (org_n % numProcesses);
        n = org_n + pad_size;

        double* values = new double[elems];
        int* rowIdx = new int[n + 1];
        int* colIdx = new int[elems];
        
        for (int i=0; i<elems; i++)
            ReadFile >> values[i];

        for (int i=0; i<org_n+1; i++)
            ReadFile >> rowIdx[i];
        // Pad matrix with zeros
        for (int i=org_n+1; i<org_n+1+pad_size; i++)
            rowIdx[i] = 0;

        for (int i=0; i<elems; i++)
            ReadFile >> colIdx[i];
        
        ReadFile.close();

        whole_A = new SparseMatrixFrag(n, pad_size, elems, values, rowIdx, colIdx, 0, n);
    } 

    // Broadcast matrix size
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    // Broadcast padding size
    MPI_Bcast(
        &pad_size,
        1,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    assert (n > 0);
    assert (pad_size >= 0);
    assert (numProcesses <= n);
    assert (n % numProcesses == 0);  // Check if we padded the matrix correctly

    int groupSize = numProcesses/c;

    // Distribute chunks of A over all processes
    if (myRank == ROOT_PROCESS) {
        std::vector<SparseMatrixFrag*> chunks = whole_A->chunk(groupSize);
        for (int processNum=1; processNum<numProcesses; processNum++){
            SparseMatrixFrag* chunk = chunks[processNum%groupSize];
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

            if (chunkNumElems > 0) {
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
        }

        // Initialize chunk of ROOT process
        A = chunks[0];
        
        // Delete temporary chunks
        for (size_t i=1; i<chunks.size(); i++)
            delete(chunks[i]);
        // ROOT process no longer needs to store the whole matrix A
        delete(whole_A);
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

        int firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n, 0 /*round*/, c);
        int lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n, 0 /*round*/, c);

        if (chunkNumElems > 0) {
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

            A = new SparseMatrixFrag(n, pad_size, chunkNumElems, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
        }
        else {
            // Create empty matrix
            A = new SparseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl);
        }
    }

    // Generate fragment of dense matrix
    firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n);
    lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n);
    B = new DenseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl, seed);

    // ColA algorithm
    for (int iteration = 0; iteration < e; iteration++) {
        C = new DenseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl);  // seed is 0, so matrix is all zeros
        for (int round=1; round<=groupSize; round++) {
            multiplyColA(A, B, C);

            A = shiftColA(A, myRank, numProcesses, round, c);
        }
        delete(B);
        B = C;
    }

    // Show result
    whole_C = gatherResultColA(myRank, numProcesses, C);
    if (myRank == ROOT_PROCESS) {
        if (verbose)
            whole_C->printout();
        if (g)
            whole_C->printout(g_val);
    }
    
    // Clean up
    MPI_Finalize();
    delete(A);
    delete(B);
    if (myRank == ROOT_PROCESS)
        delete(whole_C);
}