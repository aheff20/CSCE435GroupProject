# CSCE 435 Group project

## 1. Group members:
1. Aidan Heffron
2. Miguel Garcia
3. Joey Quismorio
4. James Evans

## 2. Communication
Our group will communicate with each other through the Group Chat we have created with IMessage. Deadlines and tasks will be discussed well in advance (at least 48 hours for smaller deadlines, 96 for larger ones), giving each team member time to implement their scheduled tasks. Any conflicts should be communicated ASAP to give the team time to adjust. 

---

## 3. _due 10/25_ Project topic
- Choose 3+ parallel sorting algorithms, implement in MPI and CUDA.  Examine and compare performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance

## 4. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
The project will include the following algorithms and architectures:

- Merge Sort (CUDA)
- Merge Sort (MPI on each core)
- Radix Sort (CUDA)
- Radix Sort (MPI on each core)
- Quick Sort (CUDA)
- Quick Sort (MPI on each core)

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
