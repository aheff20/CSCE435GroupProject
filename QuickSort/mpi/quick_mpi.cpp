#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string>
#include <cmath>

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

float *global_array;
int *global_values_to_send;
int *global_counts;

int compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}


float select_pivot(float *values, int size) {
    return values[size-1];
}

int find_partner(int rankid, int depth) {

    int groupSize = (int) (numTasks / (pow(2, depth)));
    int partnerGroup = (int)(rankid / groupSize);
    int partnerOffset = rankid % groupSize;

    int partnerId;

    if (partnerOffset < (int)(groupSize / 2)) {
        partnerId = (int)(rankid + groupSize / 2);
    } else {
        partnerId = (int)(rankid - groupSize / 2);
    }

    return partnerId;

}

bool select_next_pivot(int rankid, int depth) {

    for(int i = 0; i < numTasks; i += numTasks/(pow(2,depth))){
        if(i == rankid) {
            return true;
        }
    }

    return false;

}

int determine_source(int rankid, int depth) {

    int source = 0;
    for(int i = 0; i < numTasks; i += numTasks/(pow(2,depth))){
        if(i < rankid) {
            source = i;
        }
    }

    return source;

}

void quickSort_step(float* local_values, int local_data_size, float pivot, int depth, int rankid) {

    if(depth >= (int)log2(numTasks)) {
        
        qsort(local_values, local_data_size, sizeof(float), compare);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Allgather(&local_data_size, 1, MPI_INT, global_counts, 1, MPI_INT, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if(rankid==0) {
            printf("FINAL COUNTS PER PROCESS:\n");
            for(int i = 0 ;i < numTasks; i++) {
                printf("%i,", global_counts[i]);
            }
            printf("\n\n");
        }

        int *global_displacements = (int*)malloc(numTasks * sizeof(int));
        global_displacements[0] = 0;
        for (int i = 1; i < numTasks; i++){
            int sum = 0;
            for (int k = i-1; k >= 0; k--){
                sum += global_counts[k];
            }
            global_displacements[i] = sum;
        }

        MPI_Gatherv(local_values, local_data_size, MPI_FLOAT, global_array, global_counts, global_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
        free(global_displacements);

        MPI_Barrier(MPI_COMM_WORLD);        
        
        return;
    }

    int partnerID = find_partner(rankid, depth);

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

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgather(&number_of_values_to_send, 1, MPI_INT, global_values_to_send, 1, MPI_INT, MPI_COMM_WORLD);    

    MPI_Barrier(MPI_COMM_WORLD);

    float* values_to_recv = new float[global_values_to_send[partnerID]];

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Send(values_to_send, number_of_values_to_send, MPI_FLOAT, partnerID, 0, MPI_COMM_WORLD);
    MPI_Recv(values_to_recv, global_values_to_send[partnerID], MPI_FLOAT, partnerID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    free(values_to_send);

    float* new_local_values = new float[global_values_to_send[partnerID] + number_of_values_to_keep];
    for (int i = 0; i < global_values_to_send[partnerID]; i++){
        new_local_values[i] = values_to_recv[i];
    }
    for (int i = 0; i < number_of_values_to_keep; i++) {
        new_local_values[i + global_values_to_send[partnerID]] = values_to_keep[i];
    }

    free(values_to_keep);
    free(values_to_recv);

    int new_local_size = number_of_values_to_keep + global_values_to_send[partnerID];

    MPI_Barrier(MPI_COMM_WORLD);

    if(depth+1 < (int)log2(numTasks)) {
        if(select_next_pivot(rankid, depth+1)) {
            pivot = select_pivot(new_local_values, new_local_size);
            for(int i = rankid + 1; i <  rankid + 2*numTasks/(2*pow(2,depth+1)); i++) {
                // printf("DEPTH=%i : P%i am sending pivot to P%i!\n", depth+1, rankid, i);
                MPI_Send(&pivot, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        } else {
            int source = determine_source(rankid, depth+1);
            // printf("DEPTH=%i : P%i my source is P%i!\n", depth+1, rankid, source);
            MPI_Recv(&pivot, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    quickSort_step(new_local_values, new_local_size, pivot, depth + 1, rankid);

    free(new_local_values);

}


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_values> <num_processes>\n", argv[0]);
        exit(1);
    }
    
    int data_size = atoi(argv[1]);

    global_values_to_send = (int*)malloc(numTasks * sizeof(int));
    global_counts = (int*)malloc(numTasks * sizeof(int)); 
    global_array = (float*)malloc(data_size * sizeof(float));

    CALI_MARK_BEGIN(data_init);
    array_fill_ascending(global_array, data_size);
    CALI_MARK_END(data_init);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rankid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);

    if (numTasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    if(rankid == 0) {
        printf("Num vals: %i\n", data_size);
        printf("Num tasks: %i\n", numTasks);
    }

    int local_data_size = data_size / numTasks;
    float *local_values = (float*)malloc(local_data_size * sizeof(float));

    MPI_Scatter(global_array, local_data_size, MPI_FLOAT, local_values, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process generate part of the array
    // CALI_MARK_BEGIN(data_init);
    // array_fill_random_no_seed(local_values, local_data_size);
    // CALI_MARK_END(data_init);

    MPI_Barrier(MPI_COMM_WORLD);

    float pivot;
    // Process 0 select the first pivot and then broadcast to each of the other processes
    if (rankid == 0) {
        
        pivot = select_pivot(local_values, local_data_size);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&pivot, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    quickSort_step(local_values, local_data_size, pivot, 0, rankid);

    free(local_values);

    if (rankid == 0) {
        CALI_MARK_BEGIN(correctness_check);
        bool correct = check_sorted(global_array, data_size);
        CALI_MARK_END(correctness_check);

        if (correct){
            printf("Array was sorted correctly!");
        }
        else{
            printf("Array was incorrectly sorted!");
        }
        free(global_array);
       // free(global_counts);
        free(global_values_to_send);
    }

    if(rankid == 0) {
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "QuickSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", data_size); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_threads", numTasks); // The number of CUDA or OpenMP threads
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }

    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    return 0;
}