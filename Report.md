# CSCE 435 Group project

## 0. Group number: 

## 1. Group members:
1. Aidan Heffron
2. Miguel Garcia
3. Joey Quismorio
4. James Evans

## 2. Communication
Our group will communicate with each other through the Group Chat we have created with IMessage. Deadlines and tasks will be discussed well in advance (at least 48 hours for smaller deadlines, 96 for larger ones), giving each team member time to implement their scheduled tasks. Any conflicts should be communicated ASAP to give the team time to adjust. 

---

## 2. Project topic
- Choose 3+ parallel sorting algorithms, implement in MPI and CUDA.  Examine and compare performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance

## 2a. Brief project description (what algorithms will you be comparing and on what architectures)
The project will include the following algorithms and architectures:

- Merge Sort (CUDA)
- Merge Sort (MPI on each core)
- Bubble Sort (CUDA)
- Bubble Sort (MPI on each core)
- Quick Sort (CUDA)
- Quick Sort (MPI on each core)

## 2b. Psuedocode for each parallel algorithm
**Bubble Sort:**

In parallel compution, bubble sort undergoes an adaptation, commonly referred to as the Odd-Even Transposition Sort. This variant is designed to optimize data handling for concurrent operations. The essence of this strategy is to orchestrate the sorting tasks such that they are staggered across different processors, thereby leveraging parallelism.

**MPI:**
```
def bubbleSort(values, local_data_size, numTasks, rankid):
    # Allocate space for the temporary array
    temp = allocate_float_array(local_data_size)

    # Loop over all phases
    for phase in range(numTasks):
        # Determine the neighbor based on the current phase and rank
        neighbor = (rankid + 1) if (phase + rankid) % 2 == 0 else (rankid - 1)

        # Only proceed if the neighbor is within valid range
        if 0 <= neighbor < numTasks:
            # Perform the send and receive operations
            MPI_Sendrecv(values, neighbor, temp, neighbor)

            # Merge the two sorted lists based on rank comparison
            if rankid < neighbor:
                # If the current rank is lower, keep the smaller elements
                for k in range(local_data_size):
                    values[k] = min(values, temp, k)
            else:
                # If the current rank is higher, keep the larger elements
                for k in reversed(range(local_data_size)):
                    values[k] = max(values, temp, k)

    # Deallocate the temporary array after use
    deallocate(temp)
```

**CUDA:**
```
def bubble_sort_step(dev_values, size, even_phase):
    idx = compute_global_index()
    i = 2 * idx + (0 if even_phase else 1)

    if even_phase:
        # Even phase: Compare elements at even indices
        if i < size - 1 - (size % 2) and dev_values[i] > dev_values[i + 1]:
            swap(dev_values[i], dev_values[i + 1])
    else:
        # Odd phase: Compare elements at odd indices
        if i < size - 1 and dev_values[i] > dev_values[i + 1]:
            swap(dev_values[i], dev_values[i + 1])

def bubble_sort(values, size):
    dev_values = allocate_device_memory(size)

    # Copy data from host to device
    copy_host_to_device(values, dev_values)

    # Calculate the number of phases needed
    major_step = size / 2
    threads_per_block = determine_threads_per_block()
    blocks = calculate_number_of_blocks(size, threads_per_block)

    # Perform the sort
    for i in range(size):
        even_phase = (i % 2) == 0
        # Launch the GPU kernel
        gpu_bubble_sort_step(dev_values, size, even_phase)

    # Copy the sorted array back to the host
    copy_device_to_host(values, dev_values)

    # Free the device memory
    free_device_memory(dev_values)
```

**Merge Sort:**

**MPI:**

**CUDA:**
```
def merge_sort_step(dev_values, temp, start, middle, end):
    i, j, k = start, middle, start

    # Merge the two halves
    while i < middle and j < end:
        if dev_values[i] < dev_values[j]:
            temp[k] = dev_values[i]
            i += 1
        else:
            temp[k] = dev_values[j]
            j += 1
        k += 1

    # Copy remaining values from the first half
    while i < middle:
        temp[k] = dev_values[i]
        i += 1
        k += 1

    # Copy remaining values from the second half
    while j < end:
        temp[k] = dev_values[j]
        j += 1
        k += 1

    # Copy merged values back to original array
    for index in range(start, end):
        dev_values[index] = temp[index]

def merge_sort(values):
    dev_values, temp = allocate_device_memory(NUM_VALS), allocate_device_memory(NUM_VALS)

    # Copy data from host to device
    copy_host_to_device(values, dev_values)

    threads_per_block = determine_threads_per_block()
    blocks = calculate_number_of_blocks()

    # Merge sort with increasing width
    width = 1
    while width < NUM_VALS:
        for i in range(0, NUM_VALS, 2 * width):
            # Calculate boundaries
            start = i
            middle = min(i + width, NUM_VALS)
            end = min(i + 2 * width, NUM_VALS)

            # Launch the GPU kernel
            gpu_merge_sort_step(dev_values, temp, start, middle, end)
            synchronize_gpu()

            width *= 2

    # Copy the sorted array back to the host
    copy_device_to_host(values, dev_values)

    # Free the device memory
    free_device_memory(dev_values)
    free_device_memory(temp)
```
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

For each algorithm and architecture, the code will test the performance of the sorting algorithm, the performance of the communication used, strong and weak scaling, etc. Algorithms implemented with MPI on each core will follow the master/worker organization, and will look something like:

```
main {
    MPI_Init();
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid)
    MPI_Comm_size(MPI_COMM_WORKLD, &numtasks)

    if(taskid == MASTER) {
        // split array up and assign sections to workers
        for destination in numworkers:
            MPI_Send(/* part to sort */);

        // receive results from workers
        for result in numworkers:
            MPI_Recv(...)

    }

    if(taskid != MASTER) {
        // receive part to sort from master
        MPI_Recv(...)

        // sort array following whatever algorithm is being used (i.e merge sort, radix sort, etc.)
        sort_array()

        // send sorted array back to master
        MPI_Send(...)

    }

    // Reduce to calculate time of worker processes for analytics
    MPI_Reduce(...)

}
```

Algorithms implemented with MPI and CUDA will follow the SIMD organization, and will look something like:

```
__global__ void function current_algorithm_step {
    // this function computes a step for whatever sorting algorithm is currently being used (i.e merge sort, radix sort, etc.)
    // it is a cuda function to run on the GPU
}

function sort_array() {
    cudaEvent_t start, stop;

    cudaMemcpy(dev_values, values, size, hostToDevice)

    for i in major_step:
        for j in minor_step:
            current_algorithm_step<<<blocks, threads>>>

    synchronize()

    cudaMemcpy(values, dev_values, size, deviceToHost)

}

main {
    fill_array()
    sort_array()
    analyze_results()
}

```

Each algorithm will have different inputs and be tested at different scales to see how it performs.

### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU 


## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
