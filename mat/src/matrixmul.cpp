#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <cassert>
#include <string.h>
#include <iostream>

int main(int argc, char * argv[]) {
    int numProcesses, myRank;
    int seed, c, e;
    char* sparse_matrix_file; 
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
    e = atoi(argv[2]);

    std::cout << "Argc: " << argc << std::endl
              << "sparse_matrix_file: " << sparse_matrix_file << std::endl
              << "seed: " << seed << std::endl
              << "c: " << std::endl
              << "e: " << e << std::endl;

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
