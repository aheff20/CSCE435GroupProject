#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

__device__ int median(int a, int b, int c) {
    int middle;
    if (a < b) {
        if (b < c) {
            middle = b;
        } else {
            if (a < c) {
                middle = c;
            } else {
                middle = a;
            }
        }
    } else {
        if (a < c) {
            middle = a;
        } else {
            if (b < c) {
                middle = c;
            } else {
                middle = b;
            }
        }
    }
    return middle;
}


__global__ void lqsort(int *data, int size, int minSize) {
    extern __shared__ int sharedData[];

    // Each block sorts a sequence
    int start = blockIdx.x * blockDim.x;
    int end = start + blockDim.x - 1;
    if (end >= size) end = size - 1;

    // Stack for sequences
    int2 stack[1024];
    int top = -1;

    // Push initial sequence
    stack[++top] = make_int2(start, end);

    while (top >= 0) {
        // Pop a sequence from the stack
        int2 seq = stack[top--];
        start = seq.x;
        end = seq.y;

        if (start < end) {
            // Choose pivot
            int pivot = median(data[start], data[(start + end) / 2], data[end]);

            // Partition
            int i = start, j = end;
            while (i <= j) {
                while (data[i] < pivot) i++;
                while (data[j] > pivot) j--;
                if (i <= j) {
                    thrust::swap(data[i], data[j]);
                    i++;
                    j--;
                }
            }

            // Push sub-sequences to stack
            if (start < j) stack[++top] = make_int2(start, j);
            if (i < end) stack[++top] = make_int2(i, end);
        }
    }

    // Possibly could includ an alternative sort for small sequences
}

void gpu_quicksort(int *data, int size, int minSize, int threads) {
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (size + threads - 1) / threads;
    lqsort<<<numBlocks, threads, threads * sizeof(int)>>>(d_data, size, minSize);

    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

bool is_sorted(int* data, int size) {
    for (int i = 1; i < size; i++) {
        if (data[i - 1] > data[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <number of threads per block> <number of values>\n", argv[0]);
        exit(1);
    }

    int THREADS = atoi(argv[1]);
    int NUM_VALS = atoi(argv[2]);
    int BLOCKS = NUM_VALS / THREADS;
    
    const int size = NUM_VALS;
    int data[size];

    // Initialize data with random values
    thrust::generate(data, data + size, rand);

    gpu_quicksort(data, size, 2, NUM_VALS); // Example minSize = 32

    if (is_sorted(data, size)) {
        std::cout << "Array is sorted." << std::endl;
    } else {
        std::cout << "Error: Array is not sorted." << std::endl;
    }

    // Print sorted data
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
