/* inspired by 
 * https://stackoverflow.com/questions/23324480
 */

#include <cblas.h>
#include <iostream> // HPC can into CPP!
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {
    const long long n{std::stoi(std::string{argv[1]})};
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist{0, 1};

    // MPI initialization
    int numProcesses, myRank;
    MPI_Init(&argc, &argv); /* intialize the library with parameters caught by the runtime */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Initialize matrices
    double* A = new double[n*n];
    double* B = new double[n*n];
    double* C = new double[n*n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i*n + j] = doubleDist(rnd);
            B[i*n + j] = doubleDist(rnd);
            C[i*n + j] = 0;
        }
    }

    // Matrix multiplication
    auto startTime = std::chrono::steady_clock::now();
    
    

    auto finishTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed{finishTime - startTime};
    std::cout << "Matrix multiplication elapsed time: "<< elapsed.count() << "[s]" << std::endl;
}

