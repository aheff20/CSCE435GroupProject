#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

long THREADS;
long BLOCKS;
long NUM_VALS;


// const char* main_time = "main_time";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* correctness_check = "correctness_check";


// data[], size, threads, blocks, 
void merge_sort(float*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(float*, float*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(float*, float*, long, long, long);






int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

//     cudaDeviceProp prop;
// cudaGetDeviceProperties(&prop, 0);  // Assuming device 0, you might want to check your device ID
// printf("Max grid dimensions: x = %d, y = %d, z = %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);


    // CALI_MARK_BEGIN(main_time);
    if (argc < 4 ) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values> <method (s/r/a/p)>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = (NUM_VALS + THREADS - 1) / THREADS;

    if (NUM_VALS % THREADS != 0) {
    fprintf(stderr, "Error: <number_of_values> must be a multiple of <threads_per_block>\n");
    exit(1);
    }

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

   
    dim3 threadsPerBlock(THREADS, 1, 1); // One-dimensional block
    dim3 blocksPerGrid(BLOCKS, 1, 1); // One-dimensional grid

    // Ensure that BLOCKS does not exceed the maximum grid size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    if (BLOCKS > prop.maxGridSize[0]) {
        fprintf(stderr, "Error: Number of blocks exceeds maximum grid size\n");
        exit(1);
    }

    threadsPerBlock.x = THREADS;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = BLOCKS;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

  

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

    CALI_MARK_END(data_init);

    // CALI_MARK_BEGIN("comp_large");
    merge_sort(values,NUM_VALS,threadsPerBlock, blocksPerGrid);


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
     adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
      adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", type_of_input); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    // adiak::value("main_time", main);
    // adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    // adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    // Finalize and clean up
    adiak::fini();
    // CALI_MARK_END(main_time);
    mgr.stop();
    mgr.flush();

    // Deallocate memory
    free(values);

    return 0;
}



void merge_sort(float* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
 
    //
    float* D_data;
    float* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
     CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMalloc((void**) &D_data, size * sizeof(float));
    cudaMalloc((void**) &D_swp, size * sizeof(float));
        CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    

    // Copy from our input list into the first array
    CALI_MARK_BEGIN(comm);
    cudaMemcpy(D_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm);
  
    //
    // Copy the thread / block info to the GPU as well
    //
    cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));


    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

  
    float* A = D_data;
    float* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;


    //
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

      

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);



        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }
     CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    cudaMemcpy(data, A, size * sizeof(float), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm);
   
    
    
    // Free the GPU memory
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    cudaFree(A);
    cudaFree(B);
        CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

  
}


__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__global__ void gpu_mergesort(float* source, float* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(float* source, float* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}


