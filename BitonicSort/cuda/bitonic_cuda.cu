/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>
#include "../../Utils/helper_functions.h"

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* correctness_check = "correctness_check";
const char* cudaMemcpy_region = "cudaMemcpy";

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{

  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      // bitonic_step_count++;
    }
  }
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);
  cudaDeviceSynchronize();

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  
  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  cali::ConfigManager mgr;
  mgr.start();

  if (argc < 4 ) {
      fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values> <method (s/r/a/p)>\n", argv[0]);
      exit(1);
  }
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  std::string type_of_input;

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

  
  bitonic_sort(values); 

  CALI_MARK_BEGIN(correctness_check);
  bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly! \n");
    }
    else{
        printf("Array was incorrectly sorted! \n");
    }
  CALI_MARK_END(correctness_check);

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "BitonicSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
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

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}
