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


const char* main_time = "main_time";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";




__global__ void merge_sort_step(float *dev_values, float *temp, int n, unsigned int width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = 2 * width * idx;
    
    if (start < n) {
        unsigned int middle = min(start + width, n);
        unsigned int end = min(start + 2 * width, n);
        unsigned int i = start;
        unsigned int j = middle;
        unsigned int k = start;

        // Perform the merge operation just as before but within a range determined by the thread
        while (i < middle && j < end) {
            if (dev_values[i] < dev_values[j]) {
                temp[k++] = dev_values[i++];
            } else {
                temp[k++] = dev_values[j++];
            }
        }
        while (i < middle) temp[k++] = dev_values[i++];
        while (j < end) temp[k++] = dev_values[j++];

        // Copy sorted elements back to original array
        for (i = start; i < end; i++) {
            dev_values[i] = temp[i];
        }
    }
}



void merge_sort(float *values) {
    float *dev_values, *temp;
    int n = NUM_VALS; // Assuming NUM_VALS is the size of the 'values' array
    size_t bytes = n * sizeof(float);
    cudaMalloc((void**)&dev_values, bytes);
    cudaMalloc((void**)&temp, bytes);


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);
     CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
   

    // dim3 blocks(BLOCKS,1);    /* Number of blocks   */
     dim3 threadsPerBlock(THREADS,1);
     
    // if (threadsPerBlock.x > 1024) {
    //     // Adjust threadsPerBlock.x to the maximum allowed if it exceeds
    //     threadsPerBlock.x = 1024;
    // }
    

    // Assume major_step and minor_step are defined elsewhere
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int width;
    for (width = 1; width < n; width *= 2) {
        //dim3 numBlocks((n + (threadsPerBlock.x * 2 * width) - 1) / (threadsPerBlock.x * 2 * width));
        //  int numBlocks = (n + 2 * width * THREADS - 1) / (2 * width * THREADS);
        
        long long totalThreads = (long long)n / (2 * width);
        int numBlocks = (totalThreads + threadsPerBlock.x - 1) / threadsPerBlock.x;

                // Check if numBlocks exceeds your GPU's limit
        // cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, 0);
        // if (numBlocks > prop.maxGridSize[0]) {
        //    numBlocks = prop.maxGridSize[0];
        // }

        merge_sort_step<<<numBlocks, threadsPerBlock>>>(dev_values, temp, n, width);
        cudaDeviceSynchronize();
       
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            exit(1);
        }

    
    }

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

   CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
     CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
 

    cudaFree(dev_values);
    cudaFree(temp);
   
}

int main(int argc, char *argv[]) {
    //CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

//     cudaDeviceProp prop;
// cudaGetDeviceProperties(&prop, 0);  // Assuming device 0, you might want to check your device ID
// printf("Max grid dimensions: x = %d, y = %d, z = %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);


    CALI_MARK_BEGIN(main_time);
    if (argc < 4 ) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values> <method (s/r/a/p)>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS/  THREADS ;

    if (NUM_VALS % THREADS != 0) {
    fprintf(stderr, "Error: <number_of_values> must be a multiple of <threads_per_block>\n");
    exit(1);
    }

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // float merge_sort_step_time = 0.0f;
    // float cudaMemcpy_host_to_device_time = 0.0f;
    // float cudaMemcpy_device_to_host_time = 0.0f;
    // int kernel_calls = 0;

    float *values = (float*)malloc(NUM_VALS * sizeof(float));
    

    const char* type_of_input;
    // Initialize data
    CALI_MARK_BEGIN(data_init);
    char method = argv[3][0]; 
        switch (method) {
        case 's': // Sorted
            array_fill_ascending(values, NUM_VALS);
            type_of_input = "sorted_array";
            break;
        case 'r': // Reverse Sorted
            array_fill_descending(values, NUM_VALS);
            type_of_input = "reversed_array";
            break;
        case 'a': // almost sorted  (perturbed)
            array_fill_ascending(values, NUM_VALS);
            perturb_array(values, NUM_VALS, 0.01);
            type_of_input = "perturbed_array";
            break;
        case 'p': // Random (default)
        default:
            array_fill_random(values, NUM_VALS);
            type_of_input = "random_array";
    }
    // array_fill_descending(values,NUM_VALS);
    // array_fill_ascending(values,NUM_VALS);
    //perturb_array(values, NUM_VALS,  .01);
    // array_fill_random(values, NUM_VALS);
    CALI_MARK_END(data_init);
   // print_array(values, NUM_VALS);

    

    // Perform merge sort
    // CALI_MARK_BEGIN("comp");
    // CALI_MARK_BEGIN("comp_large");
    merge_sort(values);
    // CALI_MARK_END("comp_large");
    // CALI_MARK_END("comp");


    CALI_MARK_BEGIN(correctness_check);
    bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly! \n");
    }
    else{
         printf("Array was incorrectly sorted! \n");
    }
    //print_array(values, NUM_VALS);
    CALI_MARK_END(correctness_check);

    

    // Output timing information
    // std::cout << "Merge Sort Step Time: " << merge_sort_step_time << " ms" << std::endl;
    // std::cout << "CUDA Memcpy Host to Device Time: " << cudaMemcpy_host_to_device_time << " ms" << std::endl;
    // std::cout << "CUDA Memcpy Device to Host Time: " << cudaMemcpy_device_to_host_time << " ms" << std::endl;
    // std::cout << "Total Kernel Calls: " << kernel_calls << std::endl;
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
    adiak::value("type_of_input", type_of_input);
    // adiak::value("main_time", main);
    // adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    // adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    // Finalize and clean up
    adiak::fini();
    CALI_MARK_END(main_time);
    mgr.stop();
    mgr.flush();

    // Deallocate memory
    free(values);

    return 0;
}





