#ifndef SORTING_HELPERS_H
#define SORTING_HELPERS_H

#include <vector>


void array_fill_random(float *arr, int length);

void array_fill_ascending(float *arr, int length);


void array_fill_descending(float *arr, int length);


bool check_sorted(const float *arr, int length);


void perturb_array(float *arr, int length, float perturbation_factor);

#endif // SORTING_HELPERS_H
