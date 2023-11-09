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


__device__ void quicksort_recursive(float *data, int left, int right) {
    if (left < right) {
        float pivot = data[(left + right) / 2];
        int pivot_index = partition(data, left, right, pivot);

        if (pivot_index > left) {
            quicksort_recursive(data, left, pivot_index - 1);
        }
        if (pivot_index < right) {
            quicksort_recursive(data, pivot_index + 1, right);
        }
    }
}

__global__ void quicksort_kernel(float *data, int left, int right) {
    int i = left + blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= right) {
        quicksort_recursive(data, left, right);
    }
}

void quicksort(float *data, int n) {
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_small);

    CALI_MARK_BEGIN(comp_large);
    quicksort_kernel<<<BLOCKS, THREADS>>>(d_data, 0, n - 1);
    CALI_MARK_END(comp_large);

    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    cudaFree(d_data);
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
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", "0"); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", "0"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    return 0;
}


