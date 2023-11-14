#ifndef SORTING_HELPERS_H
#define SORTING_HELPERS_H

#include <vector>


void array_fill_random(float *arr, int length);

void array_fill_random_no_seed(float *arr, int length);

void array_fill_ascending(float *arr, int length);


void array_fill_descending(float *arr, int length);

void array_fill_ascending_local(float *arr, int length, int rankid);


void array_fill_descending_local(float *arr, int length, int rankid, int total_length);


bool check_sorted(const float *arr, int length);


void perturb_array(float *arr, int length, float perturbation_factor);

void print_array(const float *arr, int length);

#endif // SORTING_HELPERS_H
