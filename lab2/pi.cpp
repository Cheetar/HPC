#include <omp.h>
#include <iostream>
#include <iomanip>

#define STEPS 10000
#define THREADS 64 //you can also use the OMP_NUM_THREADS environmental variable

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}

double calcPi(long n) {
    if (n < 0) {
        return 0;
    }

    return 1.0 / power(16, n)
           * (4.0 / (8 * n + 1.0)
              - 2.0 / (8 * n + 4.0)
              - 1.0 / (8 * n + 5.0)
              - 1.0/(8 * n + 6.0))
           + calcPi(n - 1); 
}

double powerParallelReduction(double x, long n) {
    if (n == 0) {
        return 1;
    }

    long totalValue = 1;
    #pragma omp parallel for reduction(*:totalValue) num_threads(THREADS)
    for (long i = 0; i < n; i++) {
        totalValue *= x;
    }
    return totalValue;
}

double powerParallelCritical(double x, long n) {
    if (n == 0) {
        return 1;
    }

    long totalValue = 1;
    for (long i = 0; i < n; i++) {
        #pragma omp critical
        {
            totalValue *= x;
        }
    }
    return totalValue;
}

double calcPiParallelReduction(long n) {
    if (n < 0) {
        return 0;
    }

    double totalValue = 0.0;
    #pragma omp parallel for reduction(+:totalValue) num_threads(THREADS)
    for (long i = 0; i < n; i++) {
        totalValue += 1.0 / powerParallelReduction(16, i)
                        * (4.0 / (8 * i + 1.0)
                            - 2.0 / (8 * i + 4.0)
                            - 1.0 / (8 * i + 5.0)
                            - 1.0/(8 * i + 6.0));
    }
    return totalValue;
}

double calcPiParallelCritical(long n) {
    if (n < 0) {
        return 0;
    }

    double totalValue = 0.0;
    for (long i = 0; i < n; i++) {
        #pragma omp atomic update
        totalValue += 1.0 / powerParallelCritical(16, i)
                        * (4.0 / (8 * i + 1.0)
                            - 2.0 / (8 * i + 4.0)
                            - 1.0 / (8 * i + 5.0)
                            - 1.0/(8 * i + 6.0));
    }
    return totalValue;
}

int main(int argc, char *argv[]) {
    std::cout << std::setprecision(10) << calcPi(STEPS) << std::endl;
    std::cout << std::setprecision(10) << calcPiParallelReduction(STEPS) << std::endl;
    std::cout << std::setprecision(10) << calcPiParallelCritical(STEPS) << std::endl;
    return 0;
}
