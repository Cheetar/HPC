#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <cassert>
#include <string.h>
#include <iostream>
#include <fstream>

const static bool DEBUG = true;

int main(int argc, char * argv[]) {
    int numProcesses, myRank, seed, c, e;
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

    // g and verbose parameters are exclusive
    assert (!(g && verbose));

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

    // Read sparse matrix A
    float tmp;
    int n, elems, d;
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

    for (int i=0; i<n; i++) {
        ReadFile >> tmp;
        std::cout << tmp << " ";
    }

    std::cout << std::endl;

    for (int i=0; i<n; i++) {
        ReadFile >> tmp;
        std::cout << tmp << " ";
    }

    ReadFile.close();

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
