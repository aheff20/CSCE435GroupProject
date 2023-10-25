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

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
The project will include the following algorithms and architectures:

- Merge Sort (MPI + CUDA)
- Merge Sort (MPI on each core)
- Radix Sort (MPI + CUDA)
- Radix Sort (MPI on each core)
- Quick Sort (MPI + CUDA)
- Quick Sort (MPI on each core)

For each algorithm and architecture, the code will test the performance of the sorting algorithm, the performance of the communication used, strong and weak scaling, etc. Algorithms implemented with MPI on each core will follow the master/worker organization, and will look something like:

```
establish_master_process()
schedule_worker_processes()
worker_process_sort()
combine_worker_results()
analyze_results()
```

Algorithms implemented with MPI and CUDA will follow the SIMD organization, and will look something like:

```
create_array()
for i in array:
    algorithm_step<<<blocks, threads>>>
analyze_results()
```

Each algorithm will have different inputs and be tested at different scales to see how it performs.