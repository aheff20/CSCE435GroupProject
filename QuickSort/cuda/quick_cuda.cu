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

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%0.3f,", array[i]);
    }
    printf("\n\n");
}


__global__ quick_sort_step(float* dev_values) {
    
}


void quick_sort(float* values, int kernel_calls) {

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

    quick_sort(values, &kernel_calls);

    bool correct = check_sorted(values,NUM_VALS);
    if (correct){
        printf("Array was sorted correctly!");
    }
    else{
         printf("Array was incorrectly sorted!");
    }


}