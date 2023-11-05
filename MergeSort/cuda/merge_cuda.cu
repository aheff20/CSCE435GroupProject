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




__global__ void merge_sort_step(float *dev_values, float *temp, unsigned int start, unsigned int middle, unsigned int end) {
    unsigned int i = start;
    unsigned int j = middle;
    unsigned int k = start;
    while (i < middle && j < end) {
        if (dev_values[i] < dev_values[j]) {
            temp[k++] = dev_values[i++];
        } else {
            temp[k++] = dev_values[j++];
        }
    }
    while (i < middle) temp[k++] = dev_values[i++];
    while (j < end) temp[k++] = dev_values[j++];

    for (i = start; i < end; i++) {
        dev_values[i] = temp[i];
    }
}


void merge_sort(float *values, float *merge_sort_step_time, float *cudaMemcpy_host_to_device_time, float *cudaMemcpy_device_to_host_time, int *kernel_calls) {
    float *dev_values, *temp;
    size_t bytes = NUM_VALS * sizeof(float);
    cudaMalloc((void**)&dev_values, bytes);
    cudaMalloc((void**)&temp, bytes);


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

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */


    // Assume major_step and minor_step are defined elsewhere
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    int width;
    for (width = 1; width < NUM_VALS; width = 2 * width) {
    for (int i = 0; i < NUM_VALS; i = i + 2 * width) {
                cudaEventRecord(start);
                merge_sort_step<<<blocks, threads>>>(dev_values, temp, i, min(i+width, NUM_VALS), min(i+2*width, NUM_VALS));
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
    cudaFree(temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    if (argc < 3 ) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values> \n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

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
   // print_array(values, NUM_VALS);

    

    // Perform merge sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    merge_sort(values, &merge_sort_step_time, &cudaMemcpy_host_to_device_time, &cudaMemcpy_device_to_host_time, &kernel_calls);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");



    bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly! \n");
    }
    else{
         printf("Array was incorrectly sorted! \n");
    }
    print_array(values, NUM_VALS);

  

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
    free(values);

    return 0;
}





