/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <sys/time.h>
#include <mpi.h>
#include "laplace-common.h"

#define OPTION_VERBOSE "--verbose"

static int const MPI_UPPER_WHITE_MESSAGE_TAG = 1;
static int const MPI_LOWER_WHITE_MESSAGE_TAG = 2;
static int const MPI_UPPER_BLACK_MESSAGE_TAG = 3;
static int const MPI_LOWER_BLACK_MESSAGE_TAG = 4;
static int const ROOT_PROCESS = 0;

static void printUsage(char const* progName) {
    std::cerr << "Usage:" << std::endl <<
              "    " << progName << " [--verbose] <N>" << std::endl <<
              "Where:" << std::endl <<
              "   <N>         The number of points in each dimension (at least 4)." << std::endl <<
              "   " << OPTION_VERBOSE << "   Prints the input and output systems." << std::endl;
}

static InputOptions parseInput(int argc, char * argv[], int numProcesses) {
    int numPointsPerDimension = 0;
    bool verbose = false;
    int errorCode = 0;

    if (argc < 2) {
        std::cerr << "ERROR: Too few arguments!" << std::endl;
        printUsage(argv[0]);
        errorCode = 1;
        MPI_Finalize();
    } else if (argc > 3) {
        std::cerr << "ERROR: Too many arguments!" << std::endl;
        printUsage(argv[0]);
        errorCode = 2;
        MPI_Finalize();
    } else {
        int argIdx = 1;

        if (argc == 3) {
            if (strncmp(argv[argIdx], OPTION_VERBOSE, strlen(OPTION_VERBOSE)) != 0) {
                std::cerr << "ERROR: Unexpected option '" << argv[argIdx] << "'!" << std::endl;
                printUsage(argv[0]);
                errorCode = 3;
                MPI_Finalize();
            }
            verbose = true;
            ++argIdx;
        }

        numPointsPerDimension = std::strtol(argv[argIdx], nullptr, 10);

        if ((numPointsPerDimension < 4) || (numProcesses > numPointsPerDimension / 2)) {
            /* If we had a smaller grid, we could use the sequential version. */
            std::cerr << "ERROR: The number of points, '"
                << argv[argIdx]
                << "', should be an iteger greater than or equal to 4; and at least 2 points per process!"
                << std::endl;
            printUsage(argv[0]);
            MPI_Finalize();
            errorCode = 4;
        }
    }

    return {numPointsPerDimension, verbose, errorCode};
}

static std::tuple<int, double> performAlgorithm(int myRank, int numProcesses, GridFragment *frag, double omega, double epsilon) {
    int startRowIncl = frag->firstRowIdxIncl + (myRank == 0 ? 1 : 0);
    int endRowExcl = frag->lastRowIdxExcl - (myRank == numProcesses - 1 ? 1 : 0);

    double maxDiff = 0;
    int numIterations = 0;
    int finished = 0;
    int *all_finished = new int[1];
    int color = 0;

    MPI_Request requests_upper[2];
    MPI_Request requests_lower[2];
    MPI_Status statuses_upper[2];
    MPI_Status statuses_lower[2];

    int *upper_white = new int[frag->gridDimension/2];
    int *lower_white = new int[frag->gridDimension/2];
    int *upper_black = new int[frag->gridDimension/2];
    int *lower_black = new int[frag->gridDimension/2];

    int *partialResults;

    if (myRank == 0) {
        partialResults = new int[numProcesses];
    }

    /* TODO: change the following code fragment */
    /* Implement asynchronous communication of neighboring elements */
    /* and computation of the grid */
    /* the following code just recomputes the appropriate grid fragment */
    /* but does not communicate the partial results */
    do {
        printf("Rank %d, iteration %d\n", myRank, numIterations);

        maxDiff = 0.0;

        // Send my white field values to upper neighbour
        if (myRank != 0) {
            for (int row=0; row<4; row++) {
                printf("Printing row of data ");
                for (int i=0; i<frag->gridDimension/2; i++) {
                    printf("%f ", frag->data[1][row][i]);
                }
                printf("\n");
            }

            MPI_Isend(
                frag->data[1][1],  // White fields in the first row
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank - 1,
                MPI_UPPER_WHITE_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_upper[0]
            );
        }

        // Send my white field values to lower neighbour 
        if (myRank != numProcesses - 1) {
            for (int row=0; row<4; row++) {
                printf("Printing row of data ");
                for (int i=0; i<frag->gridDimension/2; i++) {
                    printf("%f ", frag->data[1][row][i]);
                }
                printf("\n");
            }

            MPI_Isend(
                frag->data[1][frag->lastRowIdxExcl - 1 - frag->firstRowIdxIncl + 1],  // White fields in the last row
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank + 1,
                MPI_LOWER_WHITE_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_lower[0]
            );
        }
        
        // Receive white field data from lower neighbour
        if (myRank != numProcesses - 1) {
            MPI_Irecv(
                lower_white,
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank + 1,
                MPI_UPPER_WHITE_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_lower[1]
            );
        }

        // Receive white field data from upper neighbour
        if (myRank != 0) {
            MPI_Irecv(
                upper_white,
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank - 1,
                MPI_LOWER_WHITE_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_upper[1]
            );
        }
        
        if (myRank != 0) {
            MPI_Waitall(2, requests_upper, statuses_upper);
        }

        if (myRank != numProcesses - 1) {
            MPI_Waitall(2, requests_lower, statuses_lower);
        }

        // Update black fields on my part of the stencil
        color = 0;
        for (int rowIdx = startRowIncl; rowIdx < endRowExcl; ++rowIdx) {
            for (int colIdx = 1 + (rowIdx % 2 == color ? 1 : 0); colIdx < frag->gridDimension - 1; colIdx += 2) {
                double tmp;

                if ((rowIdx == startRowIncl) && (myRank != 0)) {
                    // Get datapoint from upper neighbour
                    tmp =
                        (upper_white[colIdx/2] +
                        GP(frag, rowIdx + 1, colIdx) +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                else if ((rowIdx == endRowExcl - 1) && (myRank != numProcesses - 1)) {
                    // Get datapoint from lower neighbour
                    tmp =
                        (GP(frag, rowIdx - 1, colIdx) +
                        lower_white[colIdx/2] +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                else {
                    tmp =
                        (GP(frag, rowIdx - 1, colIdx) +
                        GP(frag, rowIdx + 1, colIdx) +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                double diff = GP(frag, rowIdx, colIdx);
                GP(frag, rowIdx, colIdx) = (1.0 - omega) * diff + omega * tmp;
                diff = fabs(diff - GP(frag, rowIdx, colIdx));

                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

        printf("Rank %d, iteration %d, end of phase one\n", myRank, numIterations);

        //////////////////////////////////////////// SECOND PHASE //////////////////////////////////////////////

        // Send my black field values to upper neighbour
        if (myRank != 0) {
            MPI_Isend(
                frag->data[0][1],  // Black first row
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank - 1,
                MPI_UPPER_BLACK_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_upper[0]
            );
        }

        // Send my black field values to lower neighbour 
        if (myRank != numProcesses - 1) {
            MPI_Isend(
                frag->data[0][frag->lastRowIdxExcl - 1 - frag->firstRowIdxIncl + 1],  // Black last row
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank + 1,
                MPI_LOWER_BLACK_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_lower[0]
            );
        }
        
        // Receive black field data from lower neighbour
        if (myRank != numProcesses - 1) {
            MPI_Irecv(
                lower_black,
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank + 1,
                MPI_UPPER_BLACK_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_lower[1]
            );
        }

        // Receive white field data from upper neighbour
        if (myRank != 0) {
            MPI_Irecv(
                upper_black,
                frag->gridDimension/2,
                MPI_DOUBLE,
                myRank - 1,
                MPI_LOWER_BLACK_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &requests_upper[1]
            );
        }
        
        printf("Rank %d, iteration %d, second phase sent msg\n", myRank, numIterations);

        if (myRank != 0) {
            MPI_Waitall(2, requests_upper, statuses_upper);
        }

        if (myRank != numProcesses - 1) {
            MPI_Waitall(2, requests_lower, statuses_lower);
        }

        printf("Rank %d, iteration %d, 2nd phase msg received\n", myRank, numIterations);

        // Update white fields on my part of the stencil
        color = 1;
        for (int rowIdx = startRowIncl; rowIdx < endRowExcl; ++rowIdx) {
            for (int colIdx = 1 + (rowIdx % 2 == color ? 1 : 0); colIdx < frag->gridDimension - 1; colIdx += 2) {
                double tmp;

                if ((rowIdx == startRowIncl) && (myRank != 0)) {
                    // Get datapoint from upper neighbour
                    tmp =
                        (upper_black[colIdx/2] +
                        GP(frag, rowIdx + 1, colIdx) +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                else if ((rowIdx == endRowExcl - 1) && (myRank != numProcesses - 1)) {
                    // Get datapoint from lower neighbour
                    tmp =
                        (GP(frag, rowIdx - 1, colIdx) +
                        lower_black[colIdx/2] +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                else {
                    tmp =
                        (GP(frag, rowIdx - 1, colIdx) +
                        GP(frag, rowIdx + 1, colIdx) +
                        GP(frag, rowIdx, colIdx - 1) +
                        GP(frag, rowIdx, colIdx + 1)
                        ) / 4.0;
                }
                double diff = GP(frag, rowIdx, colIdx);
                GP(frag, rowIdx, colIdx) = (1.0 - omega) * diff + omega * tmp;
                diff = fabs(diff - GP(frag, rowIdx, colIdx));

                if (diff > maxDiff) {
                    maxDiff = diff;
                }
            }
        }

        printf("Rank %d, iteration %d, 2nd phase calculated\n", myRank, numIterations);

        // Root process gathers if all processes have finished
        printf("Rank %d, iteration %d, maxDiff: %f, epsilon: %f\n", myRank, numIterations, maxDiff, epsilon);
        finished = !(maxDiff > epsilon);
        MPI_Gather(
            &finished,
            1 /* just one number */,
            MPI_INT,
            partialResults,
            1 /* one number per process */,
            MPI_INT,
            ROOT_PROCESS,
            MPI_COMM_WORLD
        );

        printf("Rank %d, iteration %d, 2nd phase after gather\n", myRank, numIterations);

        if (myRank == 0) {
            all_finished[0] = 1;
            for (int i=0; i<numProcesses; i++) {
                if (partialResults[i] == 0) {
                    all_finished[0] = 0;
                }
            }
        }

        printf("Rank %d, iteration %d, 2nd phase before bcast\n", myRank, numIterations);

        // Root process broadcasts flag if the computation should continue
        MPI_Bcast(
            all_finished,
            1,  // msg length
            MPI_INT,
            ROOT_PROCESS,
            MPI_COMM_WORLD
        );
        
        printf("Rank %d, iteration %d, 2nd phase after bcast\n", myRank, numIterations);

        frag->printEntireGrid(myRank,  numProcesses);

        ++numIterations;
    //} while (!all_finished[0]);
    } while (numIterations < 5);

    delete[](upper_white);
    delete[](lower_white);
    delete[](upper_black);
    delete[](lower_black);

    /* no code changes beyond this point should be needed */

    printf("Rank %d, finished\n", myRank);

    return std::make_tuple(numIterations, maxDiff);
}

int main(int argc, char *argv[]) {
    int numProcesses;
    int myRank;
    struct timeval startTime {};
    struct timeval endTime {};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    auto inputOptions = parseInput(argc, argv, numProcesses);

    if (inputOptions.getErrorCode() != 0) {
        return inputOptions.getErrorCode();
    }

    auto numPointsPerDimension = inputOptions.getNumPointsPerDimension();
    auto isVerbose = inputOptions.isVerbose();

    double omega = Utils::getRelaxationFactor(numPointsPerDimension);
    double epsilon = Utils::getToleranceValue(numPointsPerDimension);

    auto gridFragment = new GridFragment(numPointsPerDimension, numProcesses, myRank);
    gridFragment->initialize();

    if (gettimeofday(&startTime, nullptr)) {
        gridFragment->free();
        std::cerr << "ERROR: Gettimeofday failed!" << std::endl;
        MPI_Finalize();
        return 6;
    }

    /* Start of computations. */

    auto result = performAlgorithm(myRank, numProcesses, gridFragment, omega, epsilon);

    /* End of computations. */

    if (gettimeofday(&endTime, nullptr)) {
        gridFragment->free();
        std::cerr << "ERROR: Gettimeofday failed!" << std::endl;
        MPI_Finalize();
        return 7;
    }

    double duration =
            ((double) endTime.tv_sec + ((double) endTime.tv_usec / 1000000.0)) -
            ((double) startTime.tv_sec + ((double) startTime.tv_usec / 1000000.0));

    if (myRank == 0) {
        std::cerr << "Statistics: duration(s)="
                << std::fixed
                << std::setprecision(10)
                << duration << " #iters="
                << std::get<0>(result)
                << " diff="
                << std::get<1>(result)
                << " epsilon="
                << epsilon
                << std::endl;
    }

    if (isVerbose) {
        gridFragment->printEntireGrid(myRank, numProcesses);
    }

    gridFragment->free();
    MPI_Finalize();
    return 0;
}





