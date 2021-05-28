/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <cassert>
#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    /* FIXME: implement */
    assert(numVertices % numProcesses == 0);
    int rowsPerProcess = numVertices / numProcesses;

    return myRank * rowsPerProcess;
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
            numVertices,
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
    );

    if (graph == nullptr) {
        return nullptr;
    }

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    /* FIXME: implement */
    assert(graph->numVertices % numProcesses == 0);
    int rowsPerProcess = graph->numVertices / numProcesses;

    printf("MyRank: %d\n", myRank);
    printf("numProcesses: %d\n", numProcesses);
    printf("numVertices: %d\n", numVertices);
    printf("rowsPerProcess: %d\n", rowsPerProcess);

    for (int i = myRank * rowsPerProcess; i < (myRank + 1) * rowsPerProcess; ++i) {
        initializeGraphRow(graph->data[i- (myRank * rowsPerProcess)], i, graph->numVertices);
    }

    return graph;
}

void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    MPI_Status *status;

    assert(graph->numVertices % numProcesses == 0);
    int rowsPerProcess = graph->numVertices / numProcesses;

    int *buf = new int[graph->numVertices];

    /* FIXME: implement */
    if (myRank == 0) {
        // Print my part of matrix
        for (int j=0; j < rowsPerProcess; j++) {
            printGraphRow(graph->data[j], -1 /* this number is not used*/, graph->numVertices);
        }

        // Print other parts of matrix
        for (int processNum=1; processNum < numProcesses; processNum++) {
            for (int j=0; j < rowsPerProcess; j++) {
                MPI_Recv(buf,
                         graph->numVertices,  // Data length
                         MPI_INT,
                         processNum,  // Rank of sending process
                         13,
                         MPI_COMM_WORLD,
                         status);
                printGraphRow(buf, -1 /* this number is not used*/, graph->numVertices);
                }
        }
    } else {
        for (int j=0; j < rowsPerProcess; j++) {
            MPI_Send(graph->data[j],
                     graph->numVertices,  // Data length
                     MPI_INT,
                     0,  // Rank of root process
                     13,
                     MPI_COMM_WORLD);
        }
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
