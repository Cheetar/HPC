#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char * argv[]) {
    int numProcesses, myRank;
    MPI_Status *status;

    MPI_Init(&argc,&argv); /* intialize the library with parameters caught by the runtime */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    /* Read program arguments */
    // matrixmul -f sparse_matrix_file -s seed_for_dense_matrix -c repl_group_size -e exponent [-g ge_value] [-v] [-i]

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
