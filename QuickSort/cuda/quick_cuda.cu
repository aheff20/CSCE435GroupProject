#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <curand_kernel.h>

int THREADS;
int BLOCKS;
int NUM_VALS;
#define MAX_DEPTH 500
const char* quick_sort_region = "quick_sort_region";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";

void print_elapsed(clock_t start, clock_t stop){
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

void array_fill(float *arr, int length){
    srand(time(NULL));
    CALI_MARK_BEGIN(comp_small);
    for (int i = 0; i < length; ++i) {
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
     CALI_MARK_END(comp_small);
 }

__device__ float selectPivot(float* array, int size, curandState_t* state) {
    float idx = curand(state) % size;
    return array[idx];
}

__global__ void setupRandom(curandState_t* state, unsigned long seed) {
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void parallelQuicksort(float* array, int size, curandState_t* states, int numThreads, int depth) {
   __shared__ float sharedPivot;
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockId * blockDim.x + threadId;

    // Calculate the portion of the array this thread will handle
    int chunkSize = size / (gridDim.x * blockDim.x);
    int start = globalId * chunkSize;
    int end = start + chunkSize - 1;
    if (globalId == (gridDim.x * blockDim.x) - 1) {
        end = size - 1; 
    }

    //Pivot selection and broadcasting
    if (threadId == 0) {
        sharedPivot = selectPivot(array, size, &states[id]);
    }
    __syncthreads(); //Ensure pivot selection is complete

    //Partitioning the array
    int left = start;
    int right = end;
    while (left <= right) {
        while (left <= right && array[left] <= sharedPivot) {
            left++;
        }
        while (left <= right && array[right] > sharedPivot) {
            right--;
        }
        if (left < right) {
            int temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            left++;
            right--;
        }
    }
    __syncthreads(); //Ensure all threads have finished partitioning

    // Exchange partitions among threads
if (threadId < numThreads / 2) {
    int partner = numThreads - 1 - id;

    int sendSize = left - start; // Number of elements smaller than the pivot
    int receiveSize; // Number of elements larger than the pivot, but I can't access the partner thread's data
    
    // Perform the exchange
    for (int i = 0; i < min(sendSize, receiveSize); ++i) {
        // Exchange elements at corresponding positions
        float temp = array[start + i];
        array[start + i] = array[partner * chunkSize + i];
        array[partner * chunkSize + i] = temp;
    }
}
    __syncthreads(); //Ensure all exchanges are complete

    //Recursive step
    if (depth < MAX_DEPTH) {
        int mid = (start + end) / 2;
        if (start < mid) {
            parallelQuicksort<<<BLOCKS, numThreads>>>(array, mid - start, states, numThreads, depth + 1);
        }
        if (mid + 1 < end) {
            parallelQuicksort<<<BLOCKS, numThreads>>>(array, end - mid, states, numThreads, depth + 1);
        }
    }
    
    //Final quicksort of each thread's list
    //Each thread sorts its own portion of the array.
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <number of threads per block> <number of values>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    //Allocate memory for the array on host
    float *values = (float*) malloc(NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    //Allocate memory on the device rather than host
    float *d_array;
    cudaMalloc(&d_array, NUM_VALS * sizeof(float));
    cudaMemcpy(d_array, values, NUM_VALS * sizeof(float), cudaMemcpyHostToDevice);

    //Setup for random number generation in kernel
    curandState_t *d_states;
    cudaMalloc(&d_states, numThreads * sizeof(curandState_t));
    setupRandom<<<1, numThreads>>>(d_states, time(NULL)); // Setup kernel for random number generation

    start = clock();
    parallelQuicksort<<<BLOCKS, numThreads>>>(d_array, NUM_VALS, d_states, numThreads, 0);
    stop = clock();

    cudaDeviceSynchronize();

    //Copy the sorted array back to host
    cudaMemcpy(values, d_array, NUM_VALS * sizeof(float), cudaMemcpyDeviceToHost);

    print_elapsed(start, stop);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_states);

    // Free host memory
    free(values);


    mgr.stop();
    mgr.flush();

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Quicksort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", "0"); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", "1"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    return 0;
}


