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

__device__ void swap(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(float *data, int left, int right, float pivot) {
    while (left <= right) {
        while (data[left] < pivot) left++;
        while (data[right] > pivot) right--;
        if (left <= right) {
            swap(&data[left], &data[right]);
            left++;
            right--;
        }
    }
    return left;
}


__device__ void quicksort_recursive(float *data, int left, int right, float pivot) {
     if (left < right) {
        int pivot_index = partition(data, left, right, pivot);
        if (pivot_index > left) {
            quicksort_recursive(data, left, pivot_index - 1, pivot);
        }
        if (pivot_index < right) {
            quicksort_recursive(data, pivot_index + 1, right, pivot);
        }
    }
}

__global__ void quicksort_kernel(float *data, int *leftIndices, int *rightIndices, int pivot, int blockId) {
    __shared__ int newLeftIndex, newRightIndex;

    int blockStart = leftIndices[blockId];
    int blockEnd = rightIndices[blockId];
    int threadIndex = blockStart + threadIdx.x;

    if (threadIndex <= blockEnd) {
        quicksort_recursive(data, blockStart, blockEnd, pivot);
    }

    // Assuming partitioning done correctly
    if (threadIdx.x == 0) {
        newLeftIndex = blockStart; 
        newRightIndex = blockEnd; 
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        leftIndices[blockId] = newLeftIndex;
        rightIndices[blockId] = newRightIndex;
    }
}

void quicksort(float *data, int n) {
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    int *leftIndices = (int*)malloc(BLOCKS * sizeof(int));
    int *rightIndices = (int*)malloc(BLOCKS * sizeof(int));
    int *d_leftIndices, *d_rightIndices;
    cudaMalloc(&d_leftIndices, BLOCKS * sizeof(int));
    cudaMalloc(&d_rightIndices, BLOCKS * sizeof(int));

    for (int i = 0; i < BLOCKS; ++i) {
        leftIndices[i] = 0;
        rightIndices[i] = n - 1;
    }

    cudaMemcpy(d_leftIndices, leftIndices, BLOCKS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightIndices, rightIndices, BLOCKS * sizeof(int), cudaMemcpyHostToDevice);

    bool sorted = false;
    while (!sorted) {
        sorted = true;

        for (int i = 0; i < BLOCKS; ++i) {
            if (leftIndices[i] < rightIndices[i]) {
                sorted = false;
                float pivot = data[(leftIndices[i] + rightIndices[i]) / 2];
                quicksort_kernel<<<BLOCKS, THREADS>>>(d_data, d_leftIndices, d_rightIndices, pivot, i);
                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(leftIndices, d_leftIndices, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(rightIndices, d_rightIndices, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_leftIndices);
    cudaFree(d_rightIndices);

    free(leftIndices);
    free(rightIndices);
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

    float *values = (float*) malloc(NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    start = clock();
    quicksort(values, NUM_VALS);
    stop = clock();

    print_elapsed(start, stop);

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


