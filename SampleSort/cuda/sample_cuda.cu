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

int compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%0.3f,", array[i]);
    }
    printf("\n\n");
}

__global__ void partitionAndSample(float* dev_values, int num_blocks, int local_chunk, int num_of_samples, int NUM_VALS, float* all_samples, float* pivots, int* incoming_value_count, int* final_value_count) {

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int start = threadID * local_chunk;

    if(start < NUM_VALS) {
        float *local_values = new float[local_chunk];

        // Set up partition
        for (int i = start; i < start + local_chunk; i++) {
            local_values[i - start] = dev_values[i];
        }

        // if(threadID == 0) {
        //     printf("LOCAL VALUES: \n");
        //     for(int i = 0; i < local_chunk; i++){
        //         printf("%0.3f, ", local_values[i]);
        //     }
        //     printf("\n\n");
        // }

        // Sort partition
        for (int i = 0; i < local_chunk - 1; i++) {
            for (int j = 0; j < local_chunk - i - 1; j++) {
                if (local_values[j] > local_values[j + 1]) {
                    float temp = local_values[j];
                    local_values[j] = local_values[j + 1];
                    local_values[j + 1] = temp;
                }
            }
        }

        // Select samples
        
        float* samples = new float[num_of_samples];
        for (int i = 0; i < num_of_samples; i++){
            samples[i] = local_values[i * local_chunk / num_of_samples];
        }

        // Combine to all samples
        for (int i = 0; i < num_of_samples; i++) {
            all_samples[i + threadID * num_of_samples] = samples[i];
        }

        __syncthreads();

        // first thread sorts all samples and selects pivots
        if(threadID == 0) {
            // printf("\n\nALL SAMPLES\n");
            // for(int i = 0; i < num_of_samples*num_blocks; i++){
            //     printf("%0.3f, ", all_samples[i]);
            // }
            // printf("\n\n");

            for (int i = 0; i < num_of_samples*num_blocks - 1; i++) {
                for (int j = 0; j < num_of_samples*num_blocks - i - 1; j++) {
                    if (all_samples[j] > all_samples[j + 1]) {
                        float temp = all_samples[j];
                        all_samples[j] = all_samples[j + 1];
                        all_samples[j + 1] = temp;
                    }
                }
            }  
            // printf("ALL SAMPLES SORTED \n");
            // for(int i = 0; i < num_of_samples*num_blocks; i++){
            //     printf("%0.3f, ", all_samples[i]);
            // }
            // printf("\n\n");

            for (int i = num_of_samples; i < num_blocks*num_of_samples; i+=num_of_samples) {
                pivots[(i/num_of_samples)-1] = all_samples[i];
            }

            // printf("PIVOTS\n");
            // for(int i = 0; i < num_blocks-1; i++){
            //     printf("%0.3f, ", pivots[i]);
            // }
            // printf("\n\n");

        }

        __syncthreads();

        // count displaced values
        int *localCounts = new int[num_blocks];
        int *localDisplacements = new int[num_blocks];

        for (int i = 0; i < num_blocks; i++){
            localCounts[i] = 0;
        }

        for (int i = 0; i < local_chunk; i++) {
            bool placed = false;
            for (int k = 0; k < num_blocks-1; k++) {
                if (local_values[i] < pivots[k]) {
                    localCounts[k]++;
                    placed = true;
                    break;
                }
            }
            if(!placed){
                localCounts[num_blocks-1]++;
            }
        }
        

        localDisplacements[0] = 0;
        for (int i = 1; i < num_blocks; i++){
            int sum = 0;
            for (int k = i-1; k >= 0; k--){
                sum += localCounts[k];
            }
            localDisplacements[i] = sum;
        }

        for(int i = 0; i < num_blocks; i++){
            incoming_value_count[i + threadID * num_blocks] = localCounts[i];
        } 

        __syncthreads();

        if(threadID==0){
            printf("THREAD 0 STATS\n");
            for(int i = 0; i < num_blocks-1; i++){
                printf("%0.3f, ", pivots[i]);
            }
            printf("\n");
            for(int i = 0; i < local_chunk; i++){
                printf("%0.3f, ", local_values[i]);
            }
            printf("\n");
            for(int i = 0; i < num_blocks; i++){
                printf("%i, ", localCounts[i]);
            }
            printf("\n");
            for(int i = 0; i < num_blocks; i++){
                printf("%i, ", localDisplacements[i]);
            }
            printf("\n");
            for(int i = 0; i < num_blocks*num_blocks; i++) {
                printf("%i, ", incoming_value_count[i]);
            }


            printf("\n");
        }

        __syncthreads();

        // thread 0 find out how many values each block will end up with
        if(threadID == 0) {
            for (int i = 0; i < num_blocks; i++) {
                int sum = 0;
                for (int k = i; k < num_blocks*num_blocks; k+=num_blocks){
                    sum += incoming_value_count[k];
                }
                final_value_count[i] = sum;
            }
        }

        __syncthreads();

        if(threadID == 0) {
            printf("\n\nFINAL VALUE COUNT\n");
            for(int i = 0; i < num_blocks; i++) {
                printf("%i, ", final_value_count[i]);
            }
            printf("\n\n");
        }
        // send displaced values to other blocks

        for(int i = 0; i < num_blocks; i++){

            for(int k = localDisplacements[i]; k < localDisplacements[i] + localCounts[i]; k++){ 

                int offset = k - localDisplacements[i];

                for(int j = 0; j < threadID; j++){
                    offset += incoming_value_count[j*num_blocks+i];
                }       
                
                if(i > 0) {
                    for (int n = 0; n < i; n++){
                        offset += final_value_count[n];
                    }
                }            

                dev_values[offset] = local_values[k];


            }
        
        }

        __syncthreads();

        if(threadID == 0) {
            for(int i = 0; i < NUM_VALS; i++) {
                printf("%0.3f,", dev_values[i]);
            }
            printf("\n\n");
        }

        // sort final values and place in final array

         // partition final values, sort, and place in final array
        float *final_local_values = new float[final_value_count[threadID]];

        int idx = 0;
        for(int i = 0; i < threadID; i++){
            idx += final_value_count[i];
        }

        for(int i = idx; i < idx + final_value_count[threadID]; i++) {
            final_local_values[i - idx] = dev_values[i];
        }

        for (int i = 0; i < final_value_count[threadID] - 1; i++) {
            for (int j = 0; j < final_value_count[threadID] - i - 1; j++) {
                if (final_local_values[j] > final_local_values[j + 1]) {
                    float temp = final_local_values[j];
                    final_local_values[j] = final_local_values[j + 1];
                    final_local_values[j + 1] = temp;
                }
            }
        }

        for(int i = 0; i < final_value_count[threadID]; i++) {
            dev_values[i + idx] = final_local_values[i];
        }

        __syncthreads();


    }
    

}

void sample_sort(float *values, int *kernel_calls) {
    float *dev_values;
    size_t bytes = NUM_VALS * sizeof(float);
    cudaMalloc((void**)&dev_values, bytes);

    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);

    int local_chunk = NUM_VALS / BLOCKS;
    int num_of_samples = BLOCKS > local_chunk ? local_chunk / 2 : BLOCKS;
    float* all_samples;
    cudaMalloc((void**)&all_samples, num_of_samples*BLOCKS*sizeof(float));

    float* pivots;
    cudaMalloc((void**)&pivots, (BLOCKS-1)*sizeof(float));

    int* incoming_value_count;
    cudaMalloc((void**)&incoming_value_count, (BLOCKS*BLOCKS)*sizeof(int));

    int* final_value_count;
    cudaMalloc((void**)&final_value_count, (BLOCKS)*sizeof(int));

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    partitionAndSample<<<blocks, threads>>>(dev_values, BLOCKS, local_chunk, num_of_samples, NUM_VALS, all_samples, pivots, incoming_value_count, final_value_count);
    // partitionAndSample<<<blocks, threads>>>(dev_values, local_chunk, num_of_samples, NUM_VALS, all_samples);

    cudaDeviceSynchronize();

    // float* final_samples = (float*)malloc(BLOCKS*num_of_samples * sizeof(float));
    // cudaMemcpy(final_samples, all_samples, bytes, cudaMemcpyDeviceToHost);

    // print_array(final_samples, BLOCKS*num_of_samples);

    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);


    // print_array(values, NUM_VALS);

    cudaFree(dev_values);
    cudaFree(all_samples);
    cudaFree(pivots);
    cudaFree(incoming_value_count);
    cudaFree(final_value_count);

    

}


int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <threads_per_block> <number_of_values>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    int kernel_calls = 0;
    float *values = (float*)malloc(NUM_VALS * sizeof(float));

    array_fill_random(values, NUM_VALS);
    printf("###########################################################\n");
    printf("ARRAY:\n");
    print_array(values, NUM_VALS);
    printf("###########################################################\n\n");

    sample_sort(values, &kernel_calls);

    bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly!");
    }
    else{
         printf("Array was incorrectly sorted!");
    }


}
