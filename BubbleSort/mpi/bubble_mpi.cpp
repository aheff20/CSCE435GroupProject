#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

const char* main_function = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

int compare (const void * a, const void * b)
{
    return ( *(float*)a - *(float*)b );
}

// void print_array(float* array, int size) {
//     for (int i = 0; i < size; i++){
//         printf("%0.3f,", array[i]);
//     }
//     printf("\n");
// }

// void print_iarray(int* array, int size) {
//     for (int i = 0; i < size; i++){
//         printf("%i,", array[i]);
//     }
//     printf("\n");
// }

int getNeighbor(int phase, int rank) {
    int neighbor;

    /* if it's an even phase */
    if (phase % 2 == 0) {
        /* if we are an even process */
        if (rank % 2 == 0) {
            neighbor = rank + 1;
        } else {
            neighbor = rank - 1;
        }
    } else {
        /* it's an odd phase - do the opposite */
        if (rank % 2 == 0) {
            neighbor = rank - 1;
        } else {
            neighbor = rank + 1;
        }
    }
    return neighbor;
}

void bubbleSort(float *values, int local_data_size, int numTasks, int rankid) {
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    std::sort(values, values + local_data_size); // Sort local data

    auto *temp = new float[local_data_size];
    auto *merged = new float[local_data_size * 2];

    for (int i = 0; i < numTasks; i++) {
        int neighbor = getNeighbor(i, rankid);
        
        if (neighbor < 0 || neighbor >= numTasks){
            continue;
        }

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        // Correct the MPI data types
        if (rankid % 2 == 0) {
            MPI_Send(values, local_data_size, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD);
            MPI_Recv(temp, local_data_size, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(temp, local_data_size, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(values, local_data_size, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD);
        }

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);

        // Merge and sort
        std::merge(values, values + local_data_size, temp, temp + local_data_size, merged);
        std::sort(merged, merged + local_data_size * 2);

        // Split the merged array
        if (rankid < neighbor) {
            std::copy(merged, merged + local_data_size, values);
        } else {
            std::copy(merged + local_data_size, merged + local_data_size * 2, values);
        }

        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
    }

    delete[] temp;
    delete[] merged;

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}

int main(int argc, char** argv) {
    // CALI_CXX_MARK_FUNCTION;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_values> <num_processes>\n", argv[0]);
        exit(1);
    }
    int data_size = atoi(argv[1]);

    int	numTasks,
        rankid,
        rc;

    float *global_array = nullptr;

    if (rankid == 0) {
        global_array = (float*)malloc(data_size * sizeof(float));
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rankid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);

    if (numTasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    CALI_MARK_BEGIN(main_function);
    
    int local_data_size = data_size / numTasks;
    float *values = (float*)malloc(local_data_size * sizeof(float));

    CALI_MARK_BEGIN(data_init);
    array_fill_random_no_seed(values, local_data_size);
    CALI_MARK_END(data_init);

    // localBubbleSort(values, local_data_size);

    bubbleSort(values, local_data_size, numTasks, rankid);

    MPI_Gather(values, local_data_size, MPI_FLOAT, global_array, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // print_array(global_array, data_size);

    CALI_MARK_BEGIN(correctness_check);
    
    if (rankid == 0) {
        bool correct = check_sorted(global_array, data_size);
        if (correct) {
            printf("Array was sorted correctly!\n");
        } else {
            printf("Array was incorrectly sorted!\n");
        }
    }

    CALI_MARK_END(correctness_check);

    free(values);
    free(global_array);

    CALI_MARK_END(main_function);

    if(rankid == 0) {
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "Bubble sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", float); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_threads", numTasks); // The number of CUDA or OpenMP threads
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }

    MPI_Finalize();

    return 0;
}