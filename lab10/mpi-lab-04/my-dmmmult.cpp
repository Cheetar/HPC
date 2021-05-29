/* inspired by 
 * https://stackoverflow.com/questions/23324480
 */

#include <cblas.h>
#include <iostream> // HPC can into CPP!
#include <random>
#include <chrono>
#include <cassert>
#import <mpi.h>

static int const MPI_FRONT_MESSAGE_TAG = 1;
static int const MPI_BACK_MESSAGE_TAG = 2;
static int const ROOT_PROCESS = 0;
static int const n = 100;

int main(int argc, char* argv[]) {
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist{0, 1};

    // MPI initialization
    int numProcesses, myRank;
    MPI_Init(&argc, &argv); /* intialize the library with parameters caught by the runtime */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Status status;

	int source,                /* task id of message source */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset; /* used to determine rows sent to each worker */

    double A[n][n];
    double B[n][n];
    double C[n][n];

    // Matrix multiplication
    if (myRank == ROOT_PROCESS) {
        auto startTime = std::chrono::steady_clock::now();

        // Initlize matrices
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = doubleDist(rnd);
                B[i][j] = doubleDist(rnd);
                C[i][j] = 0;
            }
        }

        assert(n%numProcesses == 0);
        averow = n/numProcesses;
        offset = 0;
        for (int dest=1; dest<=numProcesses; dest++)
        {
            rows = averow;   	
            MPI_Send(&offset, 1, MPI_INT, dest, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], rows*n, MPI_DOUBLE, dest, MPI_FRONT_MESSAGE_TAG,
                    MPI_COMM_WORLD);
            MPI_Send(&B, n*n, MPI_DOUBLE, dest, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        for (int i=1; i<=numProcesses; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, MPI_BACK_MESSAGE_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, MPI_BACK_MESSAGE_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset][0], rows*n, MPI_DOUBLE, source, MPI_BACK_MESSAGE_TAG, 
                    MPI_COMM_WORLD, &status);
        }

        auto finishTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed{finishTime - startTime};
        std::cout << "Matrix multiplication elapsed time: "<< elapsed.count() << "[s]" << std::endl;
    } else if (myRank != ROOT_PROCESS) {
        MPI_Recv(&offset, 1, MPI_INT, ROOT_PROCESS, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, ROOT_PROCESS, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, rows*n, MPI_DOUBLE, ROOT_PROCESS, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, n*n, MPI_DOUBLE, ROOT_PROCESS, MPI_FRONT_MESSAGE_TAG, MPI_COMM_WORLD, &status);

        for (int k=0; k<n; k++)
            for (int i=0; i<rows; i++)
            {
            C[i][k] = 0.0;
            for (int j=0; j<n; j++)
                C[i][k] = C[i][k] + A[i][j] * B[j][k];
            }
        MPI_Send(&offset, 1, MPI_INT, ROOT_PROCESS, MPI_BACK_MESSAGE_TAG, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, ROOT_PROCESS, MPI_BACK_MESSAGE_TAG, MPI_COMM_WORLD);
        MPI_Send(&C, rows*n, MPI_DOUBLE, ROOT_PROCESS, MPI_BACK_MESSAGE_TAG, MPI_COMM_WORLD);
    }
}

