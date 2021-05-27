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

    MPI_Status *status;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int buf_send[1] = {1};
    int buf_recv[1] = {0};

    if (myRank == 0) {
	//printf("Hello world from main process. There are %d", numProcesses);
	MPI_Send(buf_send, 1, MPI_INT, 1, 13, MPI_COMM_WORLD);
	MPI_Recv(buf_recv, 1, MPI_INT, numProcesses - 1, 13, MPI_COMM_WORLD, status);
	printf("Main node. Received %d\n", buf_recv[0]);

    }
    else {
	//printf("Hello world from %d sub process.", myRank);
	MPI_Recv(buf_recv, 1, MPI_INT, myRank - 1, 13, MPI_COMM_WORLD, status);
	buf_send[0] = buf_recv[0] + myRank;
	printf("Subprocess %d, received %d, sending %d to %d\n", myRank, buf_recv[0], buf_send[0], (myRank + 1)%numProcesses);
	MPI_Send(buf_send, 1, MPI_INT, (myRank + 1)%numProcesses, 13, MPI_COMM_WORLD);
    }

    

    //unsigned t = rand() % 5;
    //sleep(t);
    //printf("Hello world from %d/%d (slept %u s)!\n", 0, 1, t);

    MPI_Finalize(); /* mark that we've finished communicating */
    
    return 0;
}
