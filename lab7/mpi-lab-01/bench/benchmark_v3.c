/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char * argv[])
{
    MPI_Init(&argc,&argv); /* intialize the library with parameters caught by the runtime */
    
    struct timespec spec;
    srand(spec.tv_nsec); // use nsec to have a different value across different processes

    int numProcesses, myRank;
    MPI_Status *status;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    double startTime;
    double endTime;
    double executionTime;

    int length;
    int no_experiments = 30;
    int no_lengths = 7;
    int lengths[7] = {1, 10, 100, 1000, 10000, 100000, 1000000};

    // Initialize buffers
    int buf_send[1000000];
    int buf_recv[1000000];
    for (int i=0; i < 1000000; i++) {
        buf_send[i] = 0;
        buf_recv[i] = 0;
    }

    for (int i=0; i<no_lengths; i++) {
	    for (int experiment=0; experiment<no_experiments; experiment++) {

    length = lengths[i];

    // Measure execution time
    clock_gettime(CLOCK_REALTIME, &spec);
    startTime = MPI_Wtime();

    if (myRank == 0) {
	MPI_Send(buf_send, length, MPI_INT, 1, 13, MPI_COMM_WORLD);
	MPI_Recv(buf_recv, length, MPI_INT, numProcesses - 1, 13, MPI_COMM_WORLD, status);
	printf("Main node. Received data.\n");
	endTime = MPI_Wtime();
	executionTime = endTime - startTime;
	printf("%d %d %.9lf\n", experiment, length, executionTime);
    }
    else {
	MPI_Recv(buf_recv, length, MPI_INT, myRank - 1, 13, MPI_COMM_WORLD, status);
	MPI_Send(buf_send, length, MPI_INT, (myRank + 1)%numProcesses, 13, MPI_COMM_WORLD);
    }
}
}

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
