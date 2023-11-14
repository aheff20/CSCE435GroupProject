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

const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

// CUDA kernel function for bubble sort step
__global__ void bubble_sort_step(float *dev_values, int num_vals) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCount = blockDim.x * gridDim.x;

    for (int i = threadId; i < num_vals - 1; i += threadCount) {
        for (int j = 0; j < num_vals - i - 1; j++) {
            if (dev_values[j] > dev_values[j + 1]) {
                float temp = dev_values[j];
                dev_values[j] = dev_values[j + 1];
                dev_values[j + 1] = temp;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    std::string input_type = argv[3];
    BLOCKS = NUM_VALS / THREADS;

    size_t bytes = NUM_VALS * sizeof(float);

    float *values;
	float *dev_values;

    values = new float[NUM_VALS];

    cudaMalloc(&dev_values, bytes);

    // Initialize data
    CALI_MARK_BEGIN(data_init);
    if (input_type == "Sorted") {
        array_fill_ascending(values, NUM_VALS);
    } else if (input_type == "ReverseSorted") {
        array_fill_descending(values, NUM_VALS);
    } else if (input_type == "Random") {
        array_fill_random(values, NUM_VALS);
    } else if (input_type == "Perturbed") {
        array_fill_ascending(values, NUM_VALS); // First fill with sorted data
        perturb_array(values, NUM_VALS, 0.01);  // Then perturb
    }
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    
    bubble_sort_step<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
    cudaDeviceSynchronize();
    
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(correctness_check);

    bool correct = check_sorted(values, NUM_VALS);
    if (correct){
        printf("Array was sorted correctly!");
    }
    else{
         printf("Array was incorrectly sorted!");
    }
    
    CALI_MARK_END(correctness_check);

    printf("%s\n", input_type.c_str());

    // Deallocate memory
    delete[] values;
	cudaFree(dev_values);
	cudaDeviceReset();

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "BubbleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    // Finalize and clean up
    adiak::fini();

    return 0;
}
