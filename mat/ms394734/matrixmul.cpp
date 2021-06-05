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


int main(int argc, char * argv[]) {
    int numProcesses, myRank, seed, c, e;
    bool g = false;
    bool verbose = false;
    bool inner = false;
    char* sparse_matrix_file; 
    double g_val;

    /* MPI initialization */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

/*
    int *a = new int[2];
    double *b = new double[2];

    if (myRank == 0) {
        a[0] = 0;
        a[1] = 1;
        b[0] = 0.1;
        b[1] = 0.2;
    }

    const int blockLengths[2] = {2, 2};
    const MPI_Aint array_of_displacements[2] = {static_cast<MPI_Aint>(0), static_cast<MPI_Aint>(2 * sizeof(MPI_INT))};
    MPI_Datatype newtype;
    const MPI_Datatype array_of_types[2] = {MPI_INT, MPI_DOUBLE};

    MPI_Type_create_struct(
        2,
        blockLengths,
        array_of_displacements,
        array_of_types,
        &newtype
    );
    MPI_Type_commit(&newtype);
    
    MPI_Status status;

    if (myRank == 0) {
        MPI_Send(
            b,
            4,
            MPI_INT,
            1,
            13,
            MPI_COMM_WORLD
        );
    } else if (myRank == 1) {
        MPI_Recv(
            b,
            4,
            MPI_INT,
            0,
            13,
            MPI_COMM_WORLD,
            &status
        );
        std::cout << b[0] << " " << b[1] << std::endl;
    }

    MPI_Finalize();
    return 0;
    */

    /* Read program arguments */
    // Order of arguments is important!
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
    
    assert (numProcesses >= c);
    assert (numProcesses % c == 0);
    assert (!(g && verbose));  // g and verbose parameters are exclusive

    if (inner)
        innerABC(sparse_matrix_file, seed, c, e, g, g_val, verbose, myRank, numProcesses);
    else
        colA(sparse_matrix_file, seed, c, e, g, g_val, verbose, myRank, numProcesses);

    return 0;
}
