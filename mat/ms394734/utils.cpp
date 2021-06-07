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

int calcFirstColIdxIncl(int myRank, int numProcesses, int n) {
    return myRank * n/numProcesses;
}

int getFirstColIdxIncl(int myRank, int numProcesses, int n, int round, int c, bool inner) {
    int groupSize = numProcesses / c;
    int groupRank = myRank % groupSize;
    int groupNum = myRank / groupSize;
    int q = numProcesses / (c * c);
    if (inner)
        // + (groupNum * q) is initial shift in InnerABC algorithm
        // + ((groupNum+1)*groupSize) is to have non-negative value in modulo
        return calcFirstColIdxIncl(((groupRank - round - (groupNum * q) + ((groupNum+1)*groupSize)) % groupSize), groupSize, n);
    else
        return calcFirstColIdxIncl(((groupRank + round) % groupSize), groupSize, n);
}

int getLastColIdxExcl(int myRank, int numProcesses, int n, int round, int c, bool inner) {
    int groupSize = numProcesses/c;
    int groupRank = myRank % groupSize;
    int groupNum = myRank / groupSize;
    int q = numProcesses / (c * c);
    if (inner)
        // + (groupNum * q) is initial shift in InnerABC algorithm
        // + ((groupNum+1)*groupSize) is to have non-negative value in modulo
        return calcFirstColIdxIncl(((groupRank - round - (groupNum * q) + ((groupNum+1)*groupSize)) % groupSize) + 1, groupSize, n);
    else
        return calcFirstColIdxIncl(((groupRank + round) % groupSize) + 1, groupSize, n);
}

int getChunkSize(int *cache, int chunkNum) {
    return cache[chunkNum + 3];
}

int getChunkNumber(int myRank, int numProcesses, int round, int c, bool inner) {
    int groupSize = numProcesses/c;
    assert (round <= groupSize);
    if (inner) {
        int q = numProcesses / (c * c);
        int groupSize = numProcesses / c;
        int groupNum = myRank / groupSize;
        int groupRank = myRank % groupSize;
        return (groupRank - round - (q * groupNum) + (2 * groupSize)) % groupSize;
    } else
        // (+ groupSize) is to avoid negative values in modulo
        return ((myRank % groupSize) - round + groupSize) % groupSize;
}


SparseMatrixFrag::SparseMatrixFrag(int n, int pad_size, int numElems, double* values, int* rowIdx, int* colIdx, int firstColIdxIncl, int lastColIdxExcl) {
    this->n = n;
    this->pad_size = pad_size;
    this->numElems = numElems;
    this->values = values;
    this->rowIdx = rowIdx;
    this->colIdx = colIdx;
    this->firstColIdxIncl = firstColIdxIncl;
    this->lastColIdxExcl = lastColIdxExcl;
}

SparseMatrixFrag::SparseMatrixFrag(int n, int pad_size, int firstColIdxIncl, int lastColIdxExcl) {
    // Create empty sparse matrix
    this->n = n;
    this->pad_size = pad_size;
    this->numElems = 0;
    this->firstColIdxIncl = firstColIdxIncl;
    this->lastColIdxExcl = lastColIdxExcl;

    this->rowIdx = new int[n+1];

    for (int row=0; row<n+1; row++)
        this->rowIdx[row] = 0; 
}

SparseMatrixFrag::~SparseMatrixFrag() {
    if (this->numElems > 0) {
        delete(this->values);
        delete(this->rowIdx);
        delete(this->colIdx);
    } else {
        delete(this->rowIdx);
    }
}

std::vector<SparseMatrixFrag*> SparseMatrixFrag::chunk(int numChunks) {
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

        double* values = new double[numElementsInChunk];
        int* rowIdx = new int[n+1];
        int* colIdx = new int[numElementsInChunk];
        std::copy(chunkValues.begin(), chunkValues.end(), values);
        std::copy(chunkRowIdx.begin(), chunkRowIdx.end(), rowIdx);
        std::copy(chunkColIdx.begin(), chunkColIdx.end(), colIdx);

        SparseMatrixFrag *chunk = new SparseMatrixFrag(this->n, this->pad_size, numElementsInChunk, values, rowIdx, colIdx, firstColIdxIncl, lastColIdxExcl);
        chunks.push_back(chunk);
    }
    return chunks;
}

void SparseMatrixFrag::printout() {
    for (int i=0; i<this->numElems; i++)
        std::cout << this->values[i] << " ";
    std::cout << std::endl;

    for (int i=0; i<this->n + 1 - pad_size; i++)
        std::cout << this->rowIdx[i] << " ";
    std::cout << std::endl;

    for (int i=0; i<this->numElems; i++)
        std::cout << this->colIdx[i] << " ";
    std::cout << std::endl;   
}


SparseMatrixFragByRow::SparseMatrixFragByRow(int n, int pad_size, int numElems, double* values, int* rowIdx, int* colIdx, int firstRowIdxIncl, int lastRowIdxExcl) {
    this->n = n;
    this->pad_size = pad_size;
    this->numElems = numElems;
    this->values = values;
    this->rowIdx = rowIdx;
    this->colIdx = colIdx;
    this->firstRowIdxIncl = firstRowIdxIncl;
    this->lastRowIdxExcl = lastRowIdxExcl;
    this->numRows = lastRowIdxExcl - firstRowIdxIncl;
}

SparseMatrixFragByRow::SparseMatrixFragByRow(int n, int pad_size, int firstRowIdxIncl, int lastRowIdxExcl) {
    // Create empty sparse matrix
    this->n = n;
    this->pad_size = pad_size;
    this->numElems = 0;
    this->firstRowIdxIncl = firstRowIdxIncl;
    this->lastRowIdxExcl = lastRowIdxExcl;
    this->numRows = lastRowIdxExcl - firstRowIdxIncl;

    this->rowIdx = new int[this->numRows + 1];

    for (int row=firstRowIdxIncl; row<lastRowIdxExcl + 1; row++) {
        int local_row = row - firstRowIdxIncl;
        this->rowIdx[local_row] = 0; 
    }
}

SparseMatrixFragByRow::~SparseMatrixFragByRow() {
    if (this->numElems > 0) {
        delete(this->values);
        delete(this->rowIdx);
        delete(this->colIdx);
    } else {
        delete(this->rowIdx);
    }
}

std::vector<SparseMatrixFragByRow*> SparseMatrixFragByRow::chunk(int numChunks) {
    assert (this->n % numChunks == 0);
    int rowsPerChunk = this->n / numChunks;
    std::vector<SparseMatrixFragByRow*> chunks;

    for (int chunkId=0; chunkId<numChunks; chunkId++) {
        int firstRowIdxIncl = getFirstColIdxIncl(chunkId, numChunks, this->n);
        int lastRowIdxExcl = getLastColIdxExcl(chunkId, numChunks, this->n);
        std::vector<double> chunkValues;
        std::vector<int> chunkRowIdx;
        std::vector<int> chunkColIdx;
        int numElementsInChunk = 0;
        chunkRowIdx.push_back(0);
        for (int row=firstRowIdxIncl; row<lastRowIdxExcl; row++) {
            int idx = this->rowIdx[row];
            int nextIdx = this->rowIdx[row+1];
            for (int i=idx; i<nextIdx; i++) {
                numElementsInChunk++;
                chunkValues.push_back(this->values[i]);
                chunkColIdx.push_back(this->colIdx[i]);
            }
            chunkRowIdx.push_back(numElementsInChunk);
        }

        double* values = new double[numElementsInChunk];
        int* rowIdx = new int[rowsPerChunk + 1];
        int* colIdx = new int[numElementsInChunk];
        std::copy(chunkValues.begin(), chunkValues.end(), values);
        std::copy(chunkRowIdx.begin(), chunkRowIdx.end(), rowIdx);
        std::copy(chunkColIdx.begin(), chunkColIdx.end(), colIdx);

        SparseMatrixFragByRow *chunk = new SparseMatrixFragByRow(this->n, this->pad_size, numElementsInChunk, values, rowIdx, colIdx, firstRowIdxIncl, lastRowIdxExcl);
        chunks.push_back(chunk);
    }
    return chunks;
}

void SparseMatrixFragByRow::printout() {
    for (int i=0; i<this->numElems; i++)
        std::cout << this->values[i] << " ";
    std::cout << std::endl;

    for (int i=0; i<this->numRows + 1; i++)
        std::cout << this->rowIdx[i] << " ";
    std::cout << std::endl;

    for (int i=0; i<this->numElems; i++)
        std::cout << this->colIdx[i] << " ";
    std::cout << std::endl;   
}


DenseMatrixFrag::DenseMatrixFrag(int n, int pad_size, int firstColIdxIncl, int lastColIdxExcl, int seed, bool initialize) {
    this->n = n;
    this->pad_size = pad_size;
    this->firstColIdxIncl = firstColIdxIncl;
    this->lastColIdxExcl = lastColIdxExcl;
    this->numElems = n*(lastColIdxExcl - firstColIdxIncl);
    this->data = new double[n*(lastColIdxExcl - firstColIdxIncl)];
    if (initialize) {
        for (int global_col=this->firstColIdxIncl; global_col<this->lastColIdxExcl; global_col++) {
            for (int row=0; row<n; row++) {
                int local_col = global_col - this->firstColIdxIncl;
                if (row >= (n - pad_size) || global_col >= (n- pad_size))
                    this->data[local_col*n + row] = 0;
                else
                    this->data[local_col*n + row] = generate_double(seed, row, global_col);
            }
        }
    }
}

DenseMatrixFrag::~DenseMatrixFrag() {
    delete(this->data);
}

void DenseMatrixFrag::add(int row, int col, double val) {
    assert(col >= this->firstColIdxIncl && col < this->lastColIdxExcl);
    int local_col = col - this->firstColIdxIncl;
    this->data[local_col*this->n + row] += val;
}

double DenseMatrixFrag::get(int row, int col) {
    assert(col >= this->firstColIdxIncl && col < this->lastColIdxExcl);
    int local_col = col - this->firstColIdxIncl;
    return this->data[local_col*this->n + row];
}

void DenseMatrixFrag::addChunk(DenseMatrixFrag* chunk, bool opt) {
    assert (chunk->firstColIdxIncl >= this->firstColIdxIncl);
    assert (chunk->lastColIdxExcl <= this->lastColIdxExcl);
    if (opt) {
        // Optimalized data copying (but the data must be contiguous)
        int offset = this->n * (chunk->firstColIdxIncl - this->firstColIdxIncl);
        std::copy(&(chunk->data[0]), &(chunk->data[chunk->numElems]), &(this->data[offset]));
    } else {
        for (int col=chunk->firstColIdxIncl; col < chunk->lastColIdxExcl; col++) {
            for (int row=0; row<this->n; row++) {
                double val = chunk->get(row, col);
                this->add(row, col, val);
            }
        }
    }
}

void DenseMatrixFrag::printout() {
    // TODO See the description of -v in the Input/Output section. Additional instructions: white-spaces are not important (values can be separated by a single space); field formatting is not important; you should use a standard format for floating point numbers (eg. 12345.67890) with at least 5 numbers after the dot.
    std::cout << this->n - this->pad_size << " " << this->n - this->pad_size << std::endl;
    for (int row=0; row<this->n - this->pad_size; row++) {
        for (int col = this->firstColIdxIncl; col<std::min(lastColIdxExcl, this->n - this->pad_size); col++) {
            int local_col = col - this->firstColIdxIncl;
            std::cout << data[local_col*n + row] << " ";
        }
        std::cout << std::endl;
    }
}

int DenseMatrixFrag::getNumberOfGreaterThan(double th) {
    // Returns the number of elements greater *or equal* value th
    int numElems = 0;
    for (int row=0; row<this->n - this->pad_size; row++) {
        for (int col = this->firstColIdxIncl; col<std::min(lastColIdxExcl, this->n - this->pad_size); col++) {
            int local_col = col - this->firstColIdxIncl;
            if (data[local_col*n + row] >= th)
                numElems++;
        }
    }
    return numElems;
}
