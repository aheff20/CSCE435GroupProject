#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

int THREADS;
int BLOCKS;
int NUM_VALS;
int inputType = 0;

const char* data_init = "data_init";
const char* sample_sort_region = "sample_sort_region";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

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
    printf("\n\n");
}

void select_pivots(float* samples, float* pivots, int samples_size, int num_of_samples) {
    for (int i = num_of_samples; i < samples_size; i += num_of_samples) {
        pivots[(i/num_of_samples)-1] = all_samples[i];
    }
}

void compute_final_counts(int* incoming_values, int* final_counts, int num_blocks) {
    for (int i = 0; i < num_blocks; i++) {
        int sum = 0;
        for (int k = i; k < num_blocks*num_blocks; k+=num_blocks){
            sum += incoming_values[k];
        }
        final_counts[i] = sum;
    }
}

__global__ void partitionAndSample(float* dev_values, int num_vals, int local_chunk, float* all_samples, int num_of_samples) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int start = threadID * local_chunk;

    if (start < num_vals) {
        // Sort portion of dev_values
        for (int i = 0; i < local_chunk - 1; i++) {
            for (int j = start; j < start + local_chunk - i - 1; j++) {
                if (dev_values[j] > dev_values[j + 1]) {
                    float temp = dev_values[j];
                    dev_values[j] = dev_values[j + 1];
                    dev_values[j + 1] = temp;
                }
            }
        }

        // Select samples and place in all samples 
        for (int i = 0; i < num_of_samples; i++){
            all_samples[i + threadID * num_of_samples] = local_values[i * local_chunk / num_of_samples];
        }
    }
}

__global__ void findDisplacements(float* dev_values, int num_vals, int num_blocks, int local_chunk, float* pivots, int* incoming_value_count, int* displacements) {
    // based on pivots, count how many values need to be sent to each other block
    // incoming_value_count is an array of BLOCKS * num_of_samples
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int start = threadID * local_chunk;

    if (start < num_vals) {
        // Grab data from partition
        float* local_values = new float[local_chunk];
        for (int i = start; i < start + local_chunk; i++) {
            local_values[i - start] = dev_values[i];
        }

        // Init local counters
        int *localCounts = new int[num_blocks];
        int *localDisplacements = new int[num_blocks];

        for (int i = 0; i < num_blocks; i++){
            localCounts[i] = 0;
        }

        // Determine where each value in the partion should go based off pivots
        for (int i = 0; i < local_chunk; i++) {
            bool placed = false;
            for (int k = 0; k < num_blocks-1; k++) {
                if (local_values[i] < pivots[k]) {
                    localCounts[k]++;
                    placed = true;
                    break;
                }
            }
            if(!placed){
                localCounts[num_blocks-1]++;
            }
        }
        
        // Calculate displacements from local counts
        localDisplacements[0] = 0;
        for (int i = 1; i < num_blocks; i++){
            int sum = 0;
            for (int k = i-1; k >= 0; k--){
                sum += localCounts[k];
            }
            localDisplacements[i] = sum;
        }

        // Place local values into global arrays
        for(int i = 0; i < num_blocks; i++){
            incoming_value_count[i + threadID * num_blocks] = localCounts[i];
            displacements[i + threadID * num_blocks] = localDisplacements[i];
        }
        
    
    }


}

__global__ void sendDisplacedValues(float* final_sorted_values, float* dev_values, int num_vals, int num_blocks, int local_chunk, int *incoming_value_count, int *displacements, int *final_value_count){
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int start = threadID * local_chunk;

    if (start < num_vals) {
        // Grab data from partition
        float* local_values = new float[local_chunk];
        for (int i = start; i < start + local_chunk; i++) {
            local_values[i - start] = dev_values[i];
        }

        for(int i = 0; i < num_blocks; i++){

            for(int k = displacements[i + threadID*num_blocks]; k < displacements[i + threadID*num_blocks] + incoming_value_count[i + threadID*num_blocks]; k++){ 

                int offset = k - displacements[i + threadID*num_blocks];

                for(int j = 0; j < threadID; j++){
                    offset += incoming_value_count[j*num_blocks+i];
                }       
                
                if(i > 0) {
                    for (int n = 0; n < i; n++){
                        offset += final_value_count[n];
                    }
                }            

                final_sorted_values[offset] = local_values[k];

            }
        
        }

    }
}


__global__ void finalSort(float *final_sorted_values, int num_vals, int local_chunk, int *final_value_count) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int start = threadID * local_chunk;

    if (start < num_vals) {

        float *final_local_values = new float[final_value_count[threadID]];

        int idx = 0;
        for(int i = 0; i < threadID; i++){
            idx += final_value_count[i];
        }

        // Grab values from partition
        for(int i = idx; i < idx + final_value_count[threadID]; i++) {
            final_local_values[i - idx] = final_sorted_values[i];
        }

        // Sort values locally
        for (int i = 0; i < final_value_count[threadID] - 1; i++) {
            for (int j = 0; j < final_value_count[threadID] - i - 1; j++) {
                if (final_local_values[j] > final_local_values[j + 1]) {
                    float temp = final_local_values[j];
                    final_local_values[j] = final_local_values[j + 1];
                    final_local_values[j + 1] = temp;
                }
            }
        }

        // Place values back into original array, sorted
        for(int i = 0; i < final_value_count[threadID]; i++) {
            final_sorted_values[i + idx] = final_local_values[i];
        }

    }

}

void sample_sort(float* values, int *kernel_calls) {
    int local_chunk = NUM_VALS / BLOCKS;
    int num_of_samples = BLOCKS > local_chunk ? local_chunk / 2 : BLOCKS;

    float *dev_values;
    size_t bytes = NUM_VALS * sizeof(float);
    cudaMalloc((void**)&dev_values, bytes);

    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);

    float *all_samples;
    cudaMalloc((void**)&all_samples, BLOCKS * num_of_samples * sizeof(float));

    
    // Partition data, sort it locally, and select samples
    CALI_MARK_BEGIN(comp_large);
    partitionAndSample<<<blocks, threads>>>(dev_values, NUM_VALS, local_chunk, all_samples, num_of_samples);
    CALI_MARK_END(comp_large);

    cudaDeviceSynchronize();

    // Collect all the samples
    float *final_samples = (float*)malloc(BLOCKS * num_of_samples * sizeof(float));

    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(final_samples, all_samples, BLOCKS * num_of_samples * sizeof(float), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);


    float* pivots = (float*)malloc((BLOCKS-1) * sizeof(float));
    CALI_MARK_BEGIN(comp_small);
    // Sort all the samples, select pivots, and communicate back to devices
    qsort(final_samples, BLOCKS*num_of_samples, sizeof(float), compare);    
    select_pivots(final_samples, pivots, BLOCKS*num_of_samples, num_of_samples);
    CALI_MARK_END(comp_small);

    float *final_pivots;
    cudaMalloc((void**)&final_pivots, (BLOCKS-1) * sizeof(float));

    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(final_pivots, pivots, (BLOCKS-1) * sizeof(float), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_small);

    // Count displaced values 
    int* incoming_value_count;
    cudaMalloc((void**)&incoming_value_count, BLOCKS * BLOCKS * sizeof(int));

    int* displacements;
    cudaMalloc((void**)&displacements, BLOCKS * BLOCKS * sizeof(int));

    CALI_MARK_BEGIN(comp_large);
    findDisplacements<<<blocks, threads>>>(dev_values, NUM_VALS, BLOCKS, local_chunk, final_pivots, incoming_value_count, displacements);
    CALI_MARK_END(comp_large);

    cudaDeviceSynchronize();

    // Find out how many values will be in each block and communicate back to device
    int* incoming_values = (int*)malloc(BLOCKS * BLOCKS * sizeof(int));

    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(incoming_values, incoming_value_count, BLOCKS*BLOCKS*sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);

    int* final_value_count;
    cudaMalloc((void**)&displacements, BLOCKS * sizeof(int));

    int* final_counts = (int*)malloc(BLOCKS * sizeof(int));

    CALI_MARK_BEGIN(comp_small);
    compute_final_counts(incoming_values, final_counts, BLOCKS);
    CALI_MARK_END(comp_small);

    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(final_value_count, final_counts, BLOCKS*sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_small);

    float *final_sorted_values;
    cudaMalloc((void**)&final_sorted_values, NUM_VALS*sizeof(float));

    CALI_MARK_BEGIN(comp_large);
    // send displaced values to other blocks
    sendDisplacedValues<<<blocks, threads>>>(final_sorted_values, dev_values, NUM_VALS, BLOCKS, local_chunk, incoming_value_count, displacements, final_value_count);
    CALI_MARK_END(comp_large);

    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comp_large);
    // final sort each partition
    finalSort<<<blocks, threads>>>(final_sorted_values, NUM_VALS, local_chunk, final_value_count);
    CALI_MARK_END(comp_large);

    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comm_large);
    // Collect the final sorted array back into values
    cudaMemcpy(values, final_sorted_values, NUM_VALS, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);

    // Free data from device
    cudaFree(dev_values);
    cudaFree(final_value_count);
    cudaFree(displacements);
    cudaFree(incoming_value_count);
    cudaFree(final_pivots);
    cudaFree(all_samples);
    cudaFree(final_sorted_values);

}





int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values>\n", argv[0]);
        exit(1);
    }

    if(argc == 4) {
        inputType = atoi(argv[3]);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    int kernel_calls = 0;
    float *values = (float*)malloc(NUM_VALS * sizeof(float));
    
    CALI_MARK_BEGIN(data_init);
    array_fill_random(values, NUM_VALS);
    CALI_MARK_END(data_init);

    printf("###########################################################\n");
    printf("ARRAY:\n");
    print_array(values, NUM_VALS);
    printf("###########################################################\n\n");


    CALI_MARK_BEGIN(sample_sort_region);
    sample_sort(values, &kernel_calls);
    CALI_MARK_END(sample_sort_region);

    CALI_MARK_BEGIN(correctness_check);
    bool correct = check_sorted(values,NUM_VALS);
    CALI_MARK_END(correctness_check);
    if (correct){
        printf("Array was sorted correctly!");
    }
    else{
         printf("Array was incorrectly sorted!");
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", float); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}
