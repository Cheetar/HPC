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

const static bool DEBUG = true;
const static int ROOT_PROCESS = 0;


class SparseMatrixFrag{
    public:
        int n;
        int firstColIdxIncl;
        int lastColIdxExcl;
        double* data;
};

int getFirstColIdxIncl(int myRank, int numProcesses, int n) {
    return myRank * n/numProcesses;
}

int getLastColIdxExcl(int myRank, int numProcesses, int n) {
    return getFirstColIdxIncl(myRank + 1, numProcesses, n);
}

class DenseMatrixFrag{
    public:
        int n;  // Matrix of size nxn
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

    n = 12;

    // Root process reads and distributes sparse matrix A
    if (myRank == ROOT_PROCESS) {
        /*
        float tmp;
        int elems, d;
        std::ifstream ReadFile(sparse_matrix_file);

        ReadFile >> n;
        ReadFile >> n;  // A is a square matrix
        ReadFile >> elems;
        ReadFile >> d;


        std::cout << "n: " << n << std::endl
                    << "elems: " << elems << std::endl
                    << "d: " << d << std::endl;

        for (int i=0; i<elems; i++) {
            ReadFile >> tmp;
            std::cout << tmp << " ";
        }

        std::cout << std::endl;

        for (int i=0; i<n+1; i++) {
            ReadFile >> tmp;
            std::cout << tmp << " ";
        }

        std::cout << std::endl;

        for (int i=0; i<elems; i++) {
            ReadFile >> tmp;
            std::cout << tmp << " ";
        }
        
        ReadFile.close();
        */
    } else {
        // TODO receive fragment of matrix A
    }

    // TODO Root process broadcasts n

    // Generate fragment of dense matrix
    DenseMatrixFrag denseFrag = DenseMatrixFrag(n, myRank, numProcesses, seed);
    denseFrag.printout();

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
