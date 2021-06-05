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
        int curRow, nextRow;
        double val;
        # pragma omp for collapse(2)
        for (int row=0; row<A->n; row++) {
            for (int col=B->firstColIdxIncl; col<B->lastColIdxExcl; col++) {
                curRow = A->rowIdx[row];
                nextRow = A->rowIdx[row + 1];
                val = 0;
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

SparseMatrixFrag* shiftColA(SparseMatrixFrag* A, int* cache, int myRank, int numProcesses, int round, int c) {
    int n = A->n;
    int pad_size = A->pad_size;
    int chunkNumElems, chunkNum, bufSize;
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
    int *buf;  // buffor for receiving
    int *bufS;  // buffer for sending

    MPI_Request requestRecv;
    MPI_Status statusRecv;

    MPI_Request requestSend;
    MPI_Status statusSend;

    assert(A->numElems >= 0);

    chunkNum = getChunkNumber(myRank, numProcesses, round, c);
    chunkNumElems = getChunkSize(cache, chunkNum);

    // Receive chunk
    if (chunkNumElems > 0) {
        values = new double[chunkNumElems];
        rowIdx = new int[n + 1];
        colIdx = new int[chunkNumElems];

        bufSize = 3 * chunkNumElems + (n + 1);
        buf = new int[bufSize];

        MPI_Irecv(
            buf,
            bufSize,
            MPI_INT,
            prevProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestRecv
        );
    }

    // Send chunk
    if (A->numElems > 0) {
        // Merge 3 messages into 1
        bufSize = 3 * (A->numElems) + (n + 1);
        bufS = new int[bufSize];

        memcpy(&bufS[0], &(A->values[0]), 2 * sizeof(int) * (A->numElems));
        memcpy(&bufS[2 * (A->numElems)], &(A->colIdx[0]), sizeof(int) * (A->numElems));
        memcpy(&bufS[3 * (A->numElems)], &(A->rowIdx[0]), sizeof(int) * (n + 1));

        MPI_Isend(
            bufS,
            bufSize,
            MPI_INT,
            nextProcessNo,
            TAG,
            MPI_COMM_WORLD,
            &requestSend
        );
    }

    if (chunkNumElems > 0) {
        MPI_Wait(&requestRecv, &statusRecv);
        memcpy(values, &buf[0], 2 * sizeof(int) * chunkNumElems);
        memcpy(colIdx, &buf[2 * chunkNumElems], sizeof(int) * chunkNumElems);
        memcpy(rowIdx, &buf[3 * chunkNumElems], sizeof(int) * (n + 1));
    }
    if (A->numElems > 0)
        MPI_Wait(&requestSend, &statusSend);

    delete[](buf);
    delete[](bufS);
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
        DenseMatrixFrag* whole_C = new DenseMatrixFrag(C->n, C->pad_size, 0 /*firstColIdxIncl*/, C->n /*lastColIdxExcl*/, 0 /*seed*/, false /*dont initialize array*/); 
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
    int pad_size, chunkNumElems, org_n, n, firstColIdxIncl, lastColIdxExcl, chunkNum, bufSize;
    int* buf;

    SparseMatrixFrag* A;
    SparseMatrixFrag* whole_A;
    DenseMatrixFrag* B;
    DenseMatrixFrag* C;
    DenseMatrixFrag* whole_C;

    std::vector<SparseMatrixFrag*> chunks;

    int groupSize = numProcesses/c;
    
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
        chunks = whole_A->chunk(groupSize);
    }

    /* Cache content:
        0 - n
        1 - pad_size
        [2; groupSize + 2) - number of elements in each chunk

        BTW groupSize is equal to number of chunks
    */
    
    int *cache = new int[groupSize + 2];
    if (myRank == ROOT_PROCESS) {
        cache[0] = n;
        cache[1] = pad_size;
        for (int i=0; i < groupSize; i++)
            cache[i + 2] = chunks[i]->numElems;
    }

    // Broadcast cache
    MPI_Bcast(
        cache,
        groupSize + 2,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    n = cache[0];
    pad_size = cache[1];

    assert (n > 0);
    assert (pad_size >= 0);
    assert (numProcesses <= n);
    assert (n % numProcesses == 0);  // Check if we padded the matrix correctly

    // Distribute chunks of A over all processes
    if (myRank == ROOT_PROCESS) {
        std::vector<MPI_Request*> requests;
        std::vector<MPI_Status*> statuses;
        std::vector<int*> buffers;
        int numMessagesSent = 0;

        for (int processNum=1; processNum<numProcesses; processNum++){
            SparseMatrixFrag* chunk = chunks[processNum%groupSize];
            chunkNumElems = chunk->numElems;

            // Merge 3 messages into 1
            bufSize = 3 * chunkNumElems + (n + 1);
            buf = new int[bufSize];
            buffers.push_back(buf);

            memcpy(&buf[0], &(chunk->values[0]), 2 * sizeof(int) * chunkNumElems);
            memcpy(&buf[2 * chunkNumElems], &(chunk->colIdx[0]), sizeof(int) * chunkNumElems);
            memcpy(&buf[3 * chunkNumElems], &(chunk->rowIdx[0]), sizeof(int) * (n + 1));

            if (chunkNumElems > 0) {
                MPI_Request *request = new MPI_Request;
                MPI_Status *status = new MPI_Status;
                requests.push_back(request);
                statuses.push_back(status);
                numMessagesSent++;
                MPI_Isend(
                    buf,
                    bufSize,
                    MPI_INT,
                    processNum,
                    TAG,
                    MPI_COMM_WORLD,
                    request
                );
            }
        }

        MPI_Waitall(numMessagesSent, requests[0], statuses[0]);

        // Initialize chunk of ROOT process
        A = chunks[0];
        
        // Delete temporary chunks and buffers
        for (size_t i=1; i<chunks.size(); i++) {
            delete(chunks[i]);
            // TODO delete[](buffers[i]);
        }
        // ROOT process no longer needs to store the whole matrix A
        delete(whole_A);
    } else {
        chunkNum = myRank % groupSize;
        chunkNumElems = getChunkSize(cache, chunkNum);

        int firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n, 0 /*round*/, c);
        int lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n, 0 /*round*/, c);

        if (chunkNumElems > 0) {
            MPI_Request request;
            MPI_Status status;

            bufSize = 3 * chunkNumElems + (n + 1);
            buf = new int[bufSize];

            double* values = new double[chunkNumElems];
            int* rowIdx = new int[n + 1];
            int* colIdx = new int[chunkNumElems];

            MPI_Irecv(
                buf,
                bufSize,
                MPI_INT,
                ROOT_PROCESS,
                TAG,
                MPI_COMM_WORLD,
                &request
            ); 

            MPI_Wait(&request, &status);
            memcpy(values, &buf[0], 2 * sizeof(int) * chunkNumElems);
            memcpy(colIdx, &buf[2 * chunkNumElems], sizeof(int) * chunkNumElems);
            memcpy(rowIdx, &buf[3 * chunkNumElems], sizeof(int) * (n + 1));

            // TODO delete buf;

            A = new SparseMatrixFrag(n, pad_size, chunkNumElems, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
        }
        else {
            // Create empty matrix
            A = new SparseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl);
        }
    }

    MPI_Finalize();
    return;

    // Generate fragment of dense matrix
    firstColIdxIncl = getFirstColIdxIncl(myRank, numProcesses, n);
    lastColIdxExcl = getLastColIdxExcl(myRank, numProcesses, n);
    B = new DenseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl, seed);

    // ColA algorithm
    for (int iteration = 0; iteration < e; iteration++) {
        C = new DenseMatrixFrag(n, pad_size, firstColIdxIncl, lastColIdxExcl);  // seed is 0, so matrix is all zeros
        for (int round=1; round<=groupSize; round++) {
            multiplyColA(A, B, C);

            A = shiftColA(A, cache, myRank, numProcesses, round, c);
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