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
const char* comp = "comp";
const char* comm = "comm";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* mpi_gatherv_region = "MPI_Gatherv";
const char* mpi_gather_region = "MPI_Gather";
const char* mpi_bcast_region = "MPI_Bcast";
const char* mpi_alltoall_region = "MPI_Alltoall";
const char* mpi_alltoallv_region = "MPI_Alltoallv";
const char* mpi_allreduce_region = "MPI_Allreduce";
const char* correctness_check = "correctness_check";
const char* barrier = "MPI_Barrier";

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

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    qsort(values, local_data_size, sizeof(float), compare);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    float *samples = (float*)malloc(num_of_samples * sizeof(float));

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (int i = 0; i < num_of_samples; i++){
        samples[i] = values[i * local_data_size / num_of_samples];
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    float *all_samples = (float*)malloc(num_of_samples*numTasks * sizeof(float));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(mpi_gather_region);
    MPI_Gather(samples, num_of_samples, MPI_FLOAT, all_samples, num_of_samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_gather_region);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // at rank 0, grab all the different samples and choose pivots
    if (rankid == 0) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        qsort(all_samples, num_of_samples * numTasks, sizeof(float), compare);
        for (int i = 1; i < numTasks; ++i) {
            // choose pivots
            samples[i] = all_samples[i * numTasks + numTasks / 2];
        }
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(mpi_bcast_region);
    MPI_Bcast(samples, numTasks, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_bcast_region);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // the samples array now contains the pivots for sorting (ignore index 0 in the array)

    int *localCounts = (int*)malloc(numTasks * sizeof(int));
    int *localDisplacements = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int i = 0; i < numTasks; i++){
        localCounts[i] = 0;
    }
    
    for (int i = 0; i < local_data_size; i++) {
        bool placed = false;
        for (int k = 1; k < numTasks-1; k++) {
            if (values[i] < samples[k]) {
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
    CALI_MARK_END(comp);

    int *extCounts = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(mpi_alltoall_region);
    MPI_Alltoall(localCounts, 1, MPI_INT, extCounts, 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_alltoall_region);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    int *extDisplacements = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comp);
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
    CALI_MARK_END(comp);


    int *globalCounts = (int*)malloc(numTasks * sizeof(int));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(mpi_allreduce_region);
    MPI_Allreduce(localCounts, globalCounts, numTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_allreduce_region);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    float *sortedData = (float*)malloc(globalCounts[rankid] * sizeof(float));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_alltoallv_region);
    MPI_Alltoallv(values, localCounts, localDisplacements, MPI_FLOAT, sortedData, extCounts, extDisplacements, MPI_FLOAT, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_alltoallv_region);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    qsort(sortedData, globalCounts[rankid], sizeof(float), compare);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    // print_array(sortedData,  globalCounts[rankid]);

    int *globalDisplacements = (int*)malloc(numTasks * sizeof(int));
    CALI_MARK_BEGIN(comp);
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
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_gatherv_region);
    MPI_Gatherv(sortedData, globalCounts[rankid], MPI_FLOAT, global_array, globalCounts, globalDisplacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_gatherv_region);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    

}


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <num_values> <num_processes> <inputType>\n", argv[0]);
        exit(1);
    }
    int data_size = atoi(argv[1]);
    int inputType = atoi(argv[2]);

    int	numTasks,
        rankid,
        rc;

    float *global_array = (float*)malloc(data_size * sizeof(float));
    std::string inputTypeString;
    
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
    float *values = (float*)malloc(local_data_size * sizeof(float));
    int num_of_samples = numTasks > local_data_size ? local_data_size / 2 : numTasks;

    CALI_MARK_BEGIN(data_init);
    switch(inputType) {
        case 0:
            array_fill_random_no_seed(values, local_data_size);
            inputTypeString = "Random";
            break;
        case 1:
            array_fill_descending_local(values, local_data_size, rankid, data_size);
            inputTypeString = "ReverseSorted";
            break;
        case 2:
            array_fill_ascending_local(values, local_data_size, rankid);
            inputTypeString = "Sorted";
            break;
        case 3:
            array_fill_ascending_local(values, local_data_size, rankid);
            perturb_array(values, local_data_size, 0.01);
            inputTypeString = "1%perturbed";
            break;
        default:
            array_fill_random_no_seed(values, local_data_size);
            inputTypeString = "Random";
            break;
    }
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);
    

    sampleSort(global_array, values, rankid, local_data_size, numTasks, num_of_samples);

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
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", data_size); // The number of elements in input dataset (1000)
        adiak::value("InputType", inputTypeString); 
        adiak::value("num_threads", numTasks); // The number of CUDA or OpenMP threads
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }

    mgr.stop();
    mgr.flush();
    
    MPI_Finalize();

    return 0;
}



