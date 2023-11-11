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

const char* bubble_sort_step_region = "bubble_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

const char* main_function = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

// CUDA kernel function for bubble sort step
__global__ void bubble_sort_step(float *dev_values, int size, bool even_phase) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = 2 * idx + (even_phase ? 0 : 1); 

    if (even_phase) {
        // Even phase: Compare elements at even index with the next element
        if (i < size - 1 - (size % 2) && dev_values[i] > dev_values[i + 1]) {
            float temp = dev_values[i];
            dev_values[i] = dev_values[i + 1];
            dev_values[i + 1] = temp;
        }
    } else {
        // Odd phase: Compare elements at odd index with the next element
        if (i < size - 1 && dev_values[i] > dev_values[i + 1]) {
            float temp = dev_values[i];
            dev_values[i] = dev_values[i + 1];
            dev_values[i + 1] = temp;
        }
    }
}

// Host function to sort an array using bubble sort on the GPU
void bubbleSort(float *values, int size, int *kernel_calls) {
    float *dev_values;
    cudaMalloc((void**)&dev_values, size * sizeof(float));
    size_t bytes = size * sizeof(float);

    // Copy data from host to device

    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);

    int threads = THREADS;
    int blocks = (size + threads - 1) / threads;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    
    for (int i = 0; i < size; ++i) {
        bool even_phase = (i % 2) == 0;
        bubble_sort_step<<<blocks, threads>>>(dev_values, size, even_phase);

        (*kernel_calls)++;
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Copy the sorted array back to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Cleanup
    cudaFree(dev_values);
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN(main_function);

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    float *values = (float*)malloc(NUM_VALS * sizeof(float));
    CALI_CXX_MARK_FUNCTION;

    // Initialize data
    CALI_MARK_BEGIN("data_init");
    array_fill_random(values, NUM_VALS);
    CALI_MARK_END("data_init");

    // Declare variables for timing information
    int kernel_calls = 0;

    bubbleSort(values, NUM_VALS, &kernel_calls);

    CALI_MARK_BEGIN(correctness_check);

    bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly!");
    }
    else{
         printf("Array was incorrectly sorted!");
    }
    
    CALI_MARK_END(correctness_check);

    // Output timing information
    printf("Total Kernel Calls: %d\n", kernel_calls);

    // Adiak reporting (similar to previous example)
    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads_per_block", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("program_name", "cuda_bubble_sort");
    adiak::value("datatype_size", sizeof(float));

    // Finalize and clean up
    adiak::fini();

    // Deallocate memory
    free(values);

    CALI_MARK_END(main_function);

    return 0;
}
