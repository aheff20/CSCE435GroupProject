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

const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";

// CUDA kernel function for bubble sort step
__global__ void bubble_sort_step(float *dev_values, int size, bool even_phase) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = 2 * idx + (even_phase ? 0 : 1); // This ensures we look at even/odd pairs appropriately

    // Make sure we don't read or write out of bounds
    if (i < size - 1) {
        if (dev_values[i] > dev_values[i + 1]) {
            float temp = dev_values[i];
            dev_values[i] = dev_values[i + 1];
            dev_values[i + 1] = temp;
        }
    }
}

// Host function to sort an array using bubble sort on the GPU
void bubble_sort(float *values, int size, float *bubble_sort_step_time, float *cudaMemcpy_host_to_device_time, float *cudaMemcpy_device_to_host_time, int *kernel_calls) {
    float *dev_values;
    cudaMalloc((void**)&dev_values, size * sizeof(float));
    size_t bytes = size * sizeof(float);

    // Copy data from host to device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *cudaMemcpy_host_to_device_time += milliseconds;

    // Bubble sort is composed of NUM_VALS / 2 phases
    int major_step = size / 2; 

    int threads = THREADS;
    int blocks = (size + threads - 1) / threads;

    // Perform bubble sort with NUM_VALS / 2 phases to ensure sorting
    for (int i = 0; i < major_step; ++i) {
        bool even_phase = i % 2 == 0;
        cudaEventRecord(start);
        bubble_sort_step<<<blocks, threads>>>(dev_values, size, even_phase);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        *bubble_sort_step_time += milliseconds;
        (*kernel_calls)++;
    }

    // Copy the sorted array back to the host
    cudaEventRecord(start);
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    *cudaMemcpy_device_to_host_time += milliseconds;

    // Cleanup
    cudaFree(dev_values);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values> <blocks>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = atoi(argv[3]);

    if (NUM_VALS % (THREADS * BLOCKS) != 0) {
        fprintf(stderr, "Error: <number_of_values> must be a multiple of <threads_per_block> * <blocks>\n");
        exit(1);
    }

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
    float bubble_sort_step_time = 0.0f;
    float cudaMemcpy_host_to_device_time = 0.0f;
    float cudaMemcpy_device_to_host_time = 0.0f;
    int kernel_calls = 0;

    // Perform bubble sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bubble_sort(values, NUM_VALS, &bubble_sort_step_time, &cudaMemcpy_host_to_device_time, &cudaMemcpy_device_to_host_time, &kernel_calls);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Output timing information
    printf("Bubble Sort Step Time: %f ms\n", bubble_sort_step_time);
    printf("CUDA Memcpy Host to Device Time: %f ms\n", cudaMemcpy_host_to_device_time);
    printf("CUDA Memcpy Device to Host Time: %f ms\n", cudaMemcpy_device_to_host_time);
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
    adiak::value("bubble_sort_step_time", bubble_sort_step_time);
    adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    // Finalize and clean up
    adiak::fini();

    // Deallocate memory
    free(values);

    return 0;
}
