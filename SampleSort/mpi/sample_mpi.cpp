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
const char* sample_sort_region = "sample_sort_region";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";
const char* barrier = "mpi_barrier";

int compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%0.3f,", array[i]);
    }
    printf("\n");
}

void print_iarray(int* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%i,", array[i]);
    }
    printf("\n");
}


void sampleSort(float *global_array, float *values, int rankid, int local_data_size, int numTasks, int num_of_samples) {

    CALI_MARK_BEGIN(comp_large);
    qsort(values, local_data_size, sizeof(float), compare);
    CALI_MARK_END(comp_large);

    float *samples = (float*)malloc(num_of_samples * sizeof(float));

    CALI_MARK_BEGIN(comp_small);
    for (int i = 0; i < num_of_samples; i++){
        samples[i] = values[i * local_data_size / num_of_samples];
    }
    CALI_MARK_END(comp_small);

    float *all_samples = (float*)malloc(num_of_samples*numTasks * sizeof(float));

    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);

    CALI_MARK_BEGIN(comm_small);
    MPI_Gather(samples, num_of_samples, MPI_FLOAT, all_samples, num_of_samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);

    // at rank 0, grab all the different samples and choose pivots
    if (rankid == 0) {
        CALI_MARK_BEGIN(comp_small);
        qsort(all_samples, num_of_samples * numTasks, sizeof(float), compare);
        for (int i = 1; i < numTasks; ++i) {
            // choose pivots
            samples[i] = all_samples[i * numTasks + numTasks / 2];
        }
        CALI_MARK_END(comp_small);

        // print_array(values, local_data_size);
        // print_array(all_samples, numTasks * numTasks);
        // print_array(samples, numTasks);


    }

    CALI_MARK_BEGIN(comm_small);
    MPI_Bcast(samples, numTasks, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);

    // the samples array now contains the pivots for sorting (ignore index 0 in the array)

    int *localCounts = (int*)malloc(numTasks * sizeof(int));
    int *localDisplacements = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comp_large);
    for (int i = 0; i < numTasks; i++){
        localCounts[i] = 0;
    }
    
    for (int i = 0; i < local_data_size; i++) {
        bool placed = false;
        for (int k = 1; k < numTasks-1; k++) {
            if (local_values[i] < samples[k]) {
                localCounts[k-1]++;
                placed = true;
                break;
            }
        }
        if(!placed){
            localCounts[numTasks-1]++;
        }
    }

    localDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += localCounts[k];
        }
        localDisplacements[i] = sum;
    }
    CALI_MARK_END(comp_large);

    int *extCounts = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comm_small);
    MPI_Alltoall(localCounts, 1, MPI_INT, extCounts, 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);

    int *extDisplacements = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comp_small);
    extDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += extCounts[k];
        }
        extDisplacements[i] = sum;
    }
    CALI_MARK_END(comp_small);


    int *globalCounts = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comm_small);
    MPI_Allreduce(localCounts, globalCounts, numTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);

    // if(rankid == 0) {
    //     // print_iarray(localCounts, numTasks);
    //     printf("############################\n");
    //     print_iarray(localCounts, numTasks);
    //     print_iarray(localDisplacements, numTasks);
    //     print_iarray(extCounts, numTasks);
    //     print_iarray(extDisplacements, numTasks);
    //     print_iarray(globalCounts, numTasks);
    //     printf("############################\n");
    // }

    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);

    float *sortedData = (float*)malloc(globalCounts[rankid] * sizeof(float));

    CALI_MARK_BEGIN(comm_large);
    MPI_Alltoallv(values, localCounts, localDisplacements, MPI_FLOAT, sortedData, extCounts, extDisplacements, MPI_FLOAT, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);

    CALI_MARK_BEGIN(comp_large);
    qsort(sortedData, globalCounts[rankid], sizeof(float), compare);
    CALI_MARK_END(comp_large);
    // print_array(sortedData,  globalCounts[rankid]);

    int *globalDisplacements = (int*)malloc(numTasks * sizeof(int));
    CALI_MARK_BEGIN(comp_small);
    globalDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += globalCounts[k];
        }
        globalDisplacements[i] = sum;
    }
    CALI_MARK_END(comp_small);

    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);

    CALI_MARK_BEGIN(comm_large);
    MPI_Gatherv(sortedData, globalCounts[rankid], MPI_FLOAT, global_array, globalCounts, globalDisplacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);

    // if(rankid == 0){
    //     printf("############################\n");
    //     printf("FINAL ARRAY\n");
    //     print_array(global_array, local_data_size * numTasks);
    //     printf("############################\n");

    // }

    

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

    int	numTasks,
        rankid,
        rc;

    float *global_array = (float*)malloc(data_size * sizeof(float));

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rankid);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);

    if (numTasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    if(rankid == 0) {
        fprintf("Num vals: %i\n", data_size);
        printf("Num tasks: %i\n", numTasks);
    }
    
    int local_data_size = data_size / numTasks;
    float *values = (float*)malloc(local_data_size * sizeof(float));
    int num_of_samples = numTasks > local_chunk ? local_chunk / 2 : numTasks;

    CALI_MARK_BEGIN(data_init);
    array_fill_random_no_seed(values, local_data_size);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(sample_sort_region);
    sampleSort(global_array, values, rankid, local_data_size, numTasks, num_of_samples);
    CALI_MARK_END(sample_sort_region);

    

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

    free(values);
    free(global_array);

   
    
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

    mgr.stop();
    mgr.flush();
    
    MPI_Finalize();

    return 0;
}



