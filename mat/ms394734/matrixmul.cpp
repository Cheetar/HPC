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
#include "colA.h"
#include "innerABC.h"
#include <chrono>

int main(int argc, char * argv[]) {
    int numProcesses, myRank, seed, c, e;
    bool g = false;
    bool verbose = false;
    bool inner = false;
    char* sparse_matrix_file; 
    double g_val;

    static_assert ( sizeof(int) == 4 );
    static_assert ( sizeof(double) == 8 );

    auto start = std::chrono::steady_clock::now();

    std::ios_base::sync_with_stdio(false);

    /* MPI initialization */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    /* Read program arguments */
    assert((9 <= argc) && (argc <= 13));

    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-f") == 0) {
            assert (i + 1 < argc);
            sparse_matrix_file = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "-s") == 0) {
            assert (i + 1 < argc);
            seed = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "-c") == 0) {
            assert (i + 1 < argc);
            c = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "-e") == 0) {
            assert (i + 1 < argc);
            e = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "-g") == 0) {
            assert (i + 1 < argc);
            g = true;
            g_val = atof(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
            i++;
        } else if (strcmp(argv[i], "-i") == 0) {
            inner = true;
            i++;
        } else {
            std::cout << "Incorrect program parameters!" << std::endl;
            std::cout << "Usage: ./matrixmul -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e exponent [-g ge_value] [-v] [-i]" << std::endl;
            exit(1);
        }
    }
    
    assert (numProcesses >= c);
    assert (numProcesses % c == 0);
    assert (!(g && verbose));  // g and verbose parameters are exclusive

    if (inner)
        innerABC(sparse_matrix_file, seed, c, e, g, g_val, verbose, myRank, numProcesses);
    else
        colA(sparse_matrix_file, seed, c, e, g, g_val, verbose, myRank, numProcesses);


    auto end = std::chrono::steady_clock::now();
    if (myRank == 0)
        std::cout << "Elapsed time in milliseconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;

    return 0;
}
