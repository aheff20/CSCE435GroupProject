#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

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


void sampleSort(float *global_array, float *values, int rankid, int local_data_size, int numTasks) {

    qsort(values, local_data_size, sizeof(float), compare);

    float *samples = (float*)malloc(numTasks * sizeof(float));
    for (int i = 0; i < numTasks; i++){
        samples[i] = values[i * local_data_size / numTasks];
    }

    float *all_samples = (float*)malloc(numTasks*numTasks * sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(samples, numTasks, MPI_FLOAT, all_samples, numTasks, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // at rank 0, grab all the different samples and choose pivots
    if (rankid == 0) {
        qsort(all_samples, numTasks * numTasks, sizeof(float), compare);
        for (int i = 1; i < numTasks; ++i) {
            // choose pivots
            samples[i] = all_samples[i * numTasks + numTasks / 2];
        }

        // print_array(values, local_data_size);
        // print_array(all_samples, numTasks * numTasks);
        // print_array(samples, numTasks);


    }

    MPI_Bcast(samples, numTasks, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // the samples array now contains the pivots for sorting (ignore index 0 in the array)

    int *localCounts = (int*)malloc(numTasks * sizeof(int));
    int *localDisplacements = (int*)malloc(numTasks * sizeof(int));

    for (int i = 0; i < numTasks; i++){
        localCounts[i] = 0;
    }

    for (int i = 0; i < local_data_size; i++) {
        // int group = 0;
        if (values[i] < samples[1]) {
            localCounts[0]++;
        } else if (values[i] < samples[2]) {
            localCounts[1]++;
        } else if (values[i] < samples[3]) {
            localCounts[2]++;
        } else {
            localCounts[3]++;
        }
        // while (group < numTasks - 1 && values[i] >= samples[group]) {
        //     group++;
        // }
        // localCounts[group]++;
    }

    localDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += localCounts[k];
        }
        localDisplacements[i] = sum;
    }

    int *extCounts = (int*)malloc(numTasks * sizeof(int));
    MPI_Alltoall(localCounts, 1, MPI_INT, extCounts, 1, MPI_INT, MPI_COMM_WORLD);

    int *extDisplacements = (int*)malloc(numTasks * sizeof(int));
    extDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += extCounts[k];
        }
        extDisplacements[i] = sum;
    }


    int *globalCounts = (int*)malloc(numTasks * sizeof(int));
    MPI_Allreduce(localCounts, globalCounts, numTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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

    MPI_Barrier(MPI_COMM_WORLD);

    float *sortedData = (float*)malloc(globalCounts[rankid] * sizeof(float));
    MPI_Alltoallv(values, localCounts, localDisplacements, MPI_FLOAT, sortedData, extCounts, extDisplacements, MPI_FLOAT, MPI_COMM_WORLD);

    qsort(sortedData, globalCounts[rankid], sizeof(float), compare);
    // print_array(sortedData,  globalCounts[rankid]);

    int *globalDisplacements = (int*)malloc(numTasks * sizeof(int));
    globalDisplacements[0] = 0;
    for (int i = 1; i < numTasks; i++){
        int sum = 0;
        for (int k = i-1; k >= 0; k--){
            sum += globalCounts[k];
        }
        globalDisplacements[i] = sum;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gatherv(sortedData, globalCounts[rankid], MPI_FLOAT, global_array, globalCounts, globalDisplacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // if(rankid == 0){
    //     printf("############################\n");
    //     printf("FINAL ARRAY\n");
    //     print_array(global_array, local_data_size * numTasks);
    //     printf("############################\n");

    // }

    

}


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

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
    
    int local_data_size = data_size / numTasks;
    float *values = (float*)malloc(local_data_size * sizeof(float));

    array_fill_random_no_seed(values, local_data_size);

    sampleSort(global_array, values, rankid, local_data_size, numTasks);

    bool correct = check_sorted(global_array, data_size);

    MPI_Finalize();

    if (rankid == 0) {
        if (correct){
            printf("Array was sorted correctly!");
        }
        else{
            printf("Array was incorrectly sorted!");
        }
    }
    


    return 0;
}



