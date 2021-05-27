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
    struct timespec spec;

    MPI_Init(&argc,&argv); /* intialize the library with parameters caught by the runtime */
   
    clock_gettime(CLOCK_REALTIME, &spec);
    srand(spec.tv_nsec); // use nsec to have a different value across different processes

    int numProcesses, myRank;

    double startTime;
    double endTime;
    double executionTime;

    MPI_Status *status;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int LEN = 1000;
    int buf[LEN];
    for (int i = 0; i < LEN; i++){
	buf[i] = i;
    }

    int lengths[4] = {1, 10, 100, 1000};

    for (int i = 0; i < 4; i++) {

	int length = lengths[i];     

	for (int experiment = 0; experiment < 30; experiment++) {
		
    if (myRank == 0) {
	printf("Main node initialized\n");
	
	startTime = MPI_Wtime();

	MPI_Send(buf, length, MPI_INT, 1, 13, MPI_COMM_WORLD);
	printf("Main sent\n");
	fflush(stdout);
	MPI_Recv(buf, length, MPI_INT, 1, 13, MPI_COMM_WORLD, status);

	endTime = MPI_Wtime();
	executionTime = endTime - startTime;
	printf("%d %d %.9lf\n", experiment, length, executionTime);
    }
    else {
	if (myRank == 1) {
		printf("Secondary node\n");
		fflush(stdout);
		MPI_Recv(buf, length, MPI_INT, 0, 13, MPI_COMM_WORLD, status);
		printf("Secondary received\n");
		fflush(stdout);
		MPI_Send(buf, length, MPI_INT, 0, 13, MPI_COMM_WORLD);
	}
    }

    	}
    }

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
