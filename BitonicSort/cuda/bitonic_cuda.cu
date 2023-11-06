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

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Store results in these variables.
float effective_bandwidth_gb_s;
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;
int bitonic_step_count = 0;

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
  cudaEvent_t htd_start, htd_stop;
  cudaEventCreate(&htd_start);
  cudaEventCreate(&htd_stop);

  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);

  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(htd_start);  
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(htd_stop);
  cudaEventSynchronize(htd_stop);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, htd_start, htd_stop);
  CALI_MARK_END(cudaMemcpy_host_to_device);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  
  cudaEvent_t bit_start, bit_stop;
  cudaEventCreate(&bit_start);
  cudaEventCreate(&bit_stop);
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEventRecord(bit_start);
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      bitonic_step_count++;
    }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(bit_stop);
  cudaEventSynchronize(bit_stop);
  cudaEventElapsedTime(&bitonic_sort_step_time, bit_start, bit_stop);  
  CALI_MARK_END(bitonic_sort_step_region);


  cudaEvent_t dth_start, dth_stop;
  cudaEventCreate(&dth_start);
  cudaEventCreate(&dth_stop);
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEventRecord(dth_start);  
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(dth_stop);
  cudaEventSynchronize(dth_stop);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, dth_start, dth_stop);
  CALI_MARK_END(cudaMemcpy_device_to_host);
  
  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill_random(values, NUM_VALS);

  // start = clock();
  bitonic_sort(values); /* Inplace */
  // stop = clock();

  // print_elapsed(start, stop);

  // printf("Elapsed Time (host to device): %.3f\n", cudaMemcpy_host_to_device_time);
  // printf("Elapsed Time (device to host): %.3f\n", cudaMemcpy_device_to_host_time);
  // printf("Bitonic sort step Time (ms): %.3f\n", bitonic_sort_step_time);
  // printf("# of Bitonic Step Calls: %d\n", bitonic_step_count);


  // float bitonic_sort_time_seconds = bitonic_sort_step_time / 1000;
  // float rws = NUM_VALS * sizeof(float) * bitonic_step_count * 4;
  // effective_bandwidth_gb_s = rws / (bitonic_sort_time_seconds * 1e9);
  // printf("Effective bandwith gb/s: %.3f\n", effective_bandwidth_gb_s);

  bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly! \n");
    }
    else{
         printf("Array was incorrectly sorted! \n");
    }

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}
