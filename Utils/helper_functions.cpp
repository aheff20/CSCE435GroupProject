#include "helper_functions.h"
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>
// #include <vector>

static float random_float() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void array_fill_random(float *arr, int length) {
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < length; ++i) {
        arr[i] = random_float();
    }
}

void array_fill_random_no_seed(float *arr, int length) {
    for (int i = 0; i < length; ++i) {
        arr[i] = random_float();
    }
}

void array_fill_ascending(float *arr, int length) {
    for (int i = 0; i < length; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

void array_fill_descending(float *arr, int length) {
    for (int i = 0; i < length; ++i) {
        arr[i] = static_cast<float>(length - i - 1);
    }
}

bool check_sorted(const float *arr, int length) {
    for (int i = 1; i < length; ++i) {
        if (arr[i-1] > arr[i]) {
            std::cout << arr[i-1] << ' '<< arr[i] << ' ';
            return false; 
        }
    }
    return true;
}

void perturb_array(float *arr, int length, float perturbation_factor) {
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < length; ++i) {
   
        arr[i] += (random_float() * 2.0f - 1.0f) * perturbation_factor;
    }
}

void print_array(const float *arr, int length) {
    for (int i = 0; i < length; ++i) {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl; // End the line after printing the array
}


// void array_fill_random_descending(float *arr, int length) {
//     // First, fill the array with random values
//     for (int i = 0; i < length; ++i) {
//         arr[i] = random_float();
//     }

//     // Then sort the array in descending order using std::sort and a lambda function
//     std::sort(arr, arr + length, [](float a, float b) {
//         return a > b; // Sorting in descending order
//     });
// }