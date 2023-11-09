#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

const char* data_init = "data_init";
const char* quick_sort_region = "quick_sort_region";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";
const char* barrier = "mpi_barrier";

int	numTasks,
    rankid,
    rc;

int *global_values_to_send;

void quickSort(float* local_values, int local_data_size, float pivot, int depth, int rankid) {

    int partnerID = find_partner();

    float* values_to_keep = new float[local_data_size];
    float* values_to_send = new float[local_data_size];
    int number_of_values_to_send = 0;
    int number_of_values_to_keep = 0;

    if(partnerID > rankid) {
        for (int i = 0; i < local_data_size; i++) {
            if(local_values[i] >= pivot) {
                values_to_send[number_of_values_to_send] = local_values[i];
                number_of_values_to_send++;
            } else {
                values_to_keep[number_of_values_to_keep] = local_values[i];
                number_of_values_to_keep++;
            }
        }
    } else {
        for (int i = 0; i < local_data_size; i++) {
            if(local_values[i] < pivot) {
                values_to_send[number_of_values_to_send] = local_values[i];
                number_of_values_to_send++;
            } else {
                values_to_keep[number_of_values_to_keep] = local_values[i];
                number_of_values_to_keep++;
            }
        }
    }

    global_values_to_send[partnerID] = number_of_values_to_send;

    MPI_Barrier(MPI_COMM_WORLD);

    float* values_to_recv = new float[global_values_to_send[rankid]];

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Send(values_to_send, number_of_values_to_send, MPI_FLOAT, partnerID, 0, MPI_COMM_WORLD);
    MPI_Recv(values_to_recv, global_values_to_send[rankid], MPI_FLOAT, partnerID, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    float* new_local_values = new float[global_values_to_send[rankid] + number_of_values_to_keep];
    for (int i = 0; i < global_values_to_send[rankid]; i++){
        new_local_values[i] = values_to_recv[i];
    }
    for (int i = 0; i < number_of_values_to_keep; i++) {
        new_local_values[i + global_values_to_send[rankid]] = values_to_keep[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    

    quickSort(new_local_values, global_values_to_send[rankid] + number_of_values_to_keep, pivot, rankid);   


}


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_values> <num_processes>\n", argv[0]);
        exit(1);
    }
    
    int data_size = atoi(argv[1]);

    global_values_to_send = (int*)malloc(numTasks * sizeof(int));    

    if (numTasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rankid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);

    int local_data_size = data_size / numTasks;
    float *local_values = (float*)malloc(local_data_size * sizeof(float));

    // Each process generate part of the array
    CALI_MARK_BEGIN(data_init);
    array_fill_random_no_seed(local_values, local_data_size);
    CALI_MARK_END(data_init);

    MPI_Barrier(MPI_COMM_WORLD);

    float pivot;
    // Process 0 select the first pivot and then broadcast to each of the other processes
    if (rankid == 0) {
        
        pivot = select_pivot(local_values, local_data_size);

    }

    MPI_Bcast(pivot, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    quickSort(local_values, local_data_size, pivot, 1, rankid);



    
    MPI_Finalize();

    if (rankid == 0) {
        CALI_MARK_BEGIN(correctness_check);
        bool correct = check_sorted(global_array, data_size);
        CALI_MARK_END(sample_sort_region);

        if (correct){
            printf("Array was sorted correctly!");
        }
        else{
            printf("Array was incorrectly sorted!");
        }
    }
    
    if(rankid == 0) {
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", float); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_threads", numTasks); // The number of CUDA or OpenMP threads
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }


    return 0;
}