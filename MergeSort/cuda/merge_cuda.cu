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

const char* merge_sort_step_region = "merge_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";





__global__ void merge_sort_step(float *dev_values, int size, int width) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int start = 2 * i * width;
    
    if (start < size) {
        unsigned int middle = min(start + width, size);
        unsigned int end = min(start + 2 * width, size);
        float *temp = new float[end - start];
        
        unsigned int i = start, j = middle, k = 0;
        while (i < middle && j < end) {
            if (dev_values[i] < dev_values[j]) {
                temp[k++] = dev_values[i++];
            } else {
                temp[k++] = dev_values[j++];
            }
        }
        while (i < middle) temp[k++] = dev_values[i++];
        while (j < end) temp[k++] = dev_values[j++];

        for (i = start, k = 0; i < end; i++, k++) {
            dev_values[i] = temp[k];
        }
        delete[] temp;
    }
}


void merge_sort(float *values, int size, float *merge_sort_step_time, float *cudaMemcpy_host_to_device_time, float *cudaMemcpy_device_to_host_time, int *kernel_calls) {
    float *dev_values;
    size_t bytes = size * sizeof(float);

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

    int threads = THREADS;
    int blocks = (size + threads - 1) / threads;

    // Assume major_step and minor_step are defined elsewhere
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < major_step; ++i) {
        for (int j = 0; j < minor_step; ++j) {
            cudaEventRecord(start);
            merge_sort_step<<<blocks, threads>>>(dev_values, size, width);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&milliseconds, start, stop);
            *merge_sort_step_time += milliseconds;
            (*kernel_calls)++;
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    cudaEventRecord(start);
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    *cudaMemcpy_device_to_host_time += milliseconds;

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

    float merge_sort_step_time = 0.0f;
    float cudaMemcpy_host_to_device_time = 0.0f;
    float cudaMemcpy_device_to_host_time = 0.0f;
    int kernel_calls = 0;

    float *values = (float*)malloc(NUM_VALS * sizeof(float));
    CALI_CXX_MARK_FUNCTION;

    // Initialize data
    CALI_MARK_BEGIN("data_init");
    array_fill_random(values, NUM_VALS);
    CALI_MARK_END("data_init");

    // Declare variables for timing information
    float merge_sort_step_time = 0.0f;
    float cudaMemcpy_host_to_device_time = 0.0f;
    float cudaMemcpy_device_to_host_time = 0.0f;
    int kernel_calls = 0;

    // Perform merge sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    merge_sort(values, &merge_sort_step_time, &cudaMemcpy_host_to_device_time, &cudaMemcpy_device_to_host_time, &kernel_calls);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

  

    // Output timing information
    std::cout << "Merge Sort Step Time: " << merge_sort_step_time << " ms" << std::endl;
    std::cout << "CUDA Memcpy Host to Device Time: " << cudaMemcpy_host_to_device_time << " ms" << std::endl;
    std::cout << "CUDA Memcpy Device to Host Time: " << cudaMemcpy_device_to_host_time << " ms" << std::endl;
    std::cout << "Total Kernel Calls: " << kernel_calls << std::endl;
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
    adiak::value("program_name", "cuda_merge_sort");
    adiak::value("datatype_size", sizeof(float));
    adiak::value("merge_sort_step_time", merge_sort_step_time);
    adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    // Finalize and clean up
    adiak::fini();
    // Deallocate memory
    delete[] values;

    return 0;
}





