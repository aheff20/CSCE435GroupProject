#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

int numtasks,   /* number of tasks in partition */
    taskid,     /* a task identifier */
    numworkers, /* number of worker tasks */
    source,     /* task id of message source */
    dest,       /* task id of message destination */
    mtype;      /* message type */

float *values, *localArray;

double whole_computation_time, master_initialization_time = 0;
int height = 0;

MPI_Status status;
// Define Caliper region names
// const char *main_time = "main_time";
const char* comp_small = "comp_small";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_MPI_Barrier = "comm_MPI_Barrier";
const char *comm_large = "comm_large";
const char *comm_large_MPI_Gather = "comm_large_MPI_Gather";
const char *comm_large_MPI_Scatter = "comm_large_MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *data_init = "data_init";
const char *type_of_input;  

void merge(const float *leftArray, int leftSize, const float *rightArray, int rightSize, float *mergedArray);
int compare_floats(const void *a, const void *b);
float *mergeSortRecursive(int treeDepth, int processId, float *subArray, int subArraySize, MPI_Comm comm, float *fullArray);

int main(int argc, char *argv[])
{ // Create caliper ConfigManager object
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    // CALI_MARK_BEGIN(main_time);
    int numVals;
    if (argc == 3)
    {
        numVals = atoi(argv[1]);
    }
    else
    {
        fprintf(stderr, "\nUsage: %s<num_procs> <number_of_values>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize MPI and determine rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2)
    {
        fprintf(stderr, "Need at least two MPI tasks. Quitting...\n");
        MPI_Finalize();
        free(values);
        return EXIT_FAILURE;
    }
    numworkers = numtasks - 1;
    height = log2(numtasks);

    double start3, end3, startTime, localTime, totalTime = 0;
    start3 = MPI_Wtime();

    // Initialization
    if (taskid == MASTER)
    {
        values = (float *)malloc(numVals * sizeof(float)); // ss

        printf("merge_mpi has started with %d tasks.\n", numtasks);
        printf("merge_mpi has started with %d num vals.\n", numVals);
        printf("Initializing array...\n");
        double start, end;
        start = MPI_Wtime();
        CALI_MARK_BEGIN(data_init);

        
        // Initialize data
        char method = argv[2][0]; 
            switch (method) {
            case 's': // Sorted
                array_fill_ascending(values, numVals);
                type_of_input = "sorted_array";
                break;
            case 'r': // Reverse Sorted
                array_fill_descending(values, numVals);
                type_of_input = "reversed_array";
                break;
            case 'a': // almost sorted  (perturbed)
                array_fill_ascending(values, numVals);
                perturb_array(values, numVals, 0.01);
                type_of_input = "perturbed_array";
                break;
            case 'p': // Random (default)
            default:
                array_fill_random(values, numVals);
                type_of_input = "random_array";
        }




        array_fill_random(values, numVals);
        CALI_MARK_END(data_init);
        printf("Initialized %s  array\n",type_of_input);
        end = MPI_Wtime();
        master_initialization_time = end - start;

        // CALI_MARK_BEGIN(comm_large_MPI_Gather);
        // // Collect the sorted segments from the worker tasks
        // // ...
        // // Here you would potentially use MPI_Gather which is timed here
        // CALI_MARK_END(comm_large_MPI_Gather);
    }

    int localArraySize = numVals / numtasks;
    localArray = (float *)malloc(localArraySize * sizeof(float));
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Scatter(values, localArraySize, MPI_FLOAT, localArray, localArraySize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Merge sort
    if (taskid == 0)
    {
        double zeroStartTime = MPI_Wtime();
        // Assuming mergeSort returns a pointer to the sorted array, which is unusual.
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        values = mergeSortRecursive(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, values);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        double zeroTotalTime = MPI_Wtime() - zeroStartTime;
    }

    else
    {
         CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        double processStartTime = MPI_Wtime();
        // As a worker, you do not need to manage the global array.
        mergeSortRecursive(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, NULL);
        double processTotalTime = MPI_Wtime() - processStartTime;
    }

    // Finalize the main computation region

    end3 = MPI_Wtime();
    whole_computation_time = end3 - start3;

    localTime = MPI_Wtime() - startTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    

    // MPI_Comm worker_comm;
    // MPI_Comm_split(MPI_COMM_WORLD, 1, 1, &worker_comm);

    if (taskid == 0)
    {
        // Master Times
        printf("******************************************************\n");
        printf("Master Times:\n");
        printf("Whole Computation Time: %f \n", whole_computation_time);
        printf("Master Initialization Time: %f \n", master_initialization_time);
        printf("Master Send : %f \n", comm);
        printf("Master Recieve time: %f \n", comm_large_MPI_Gather);
        printf("\n******************************************************\n");

        // Add values to Adiak
            adiak::init(NULL);
        adiak::user();
        adiak::launchdate();
        adiak::libraries();
        adiak::cmdline();
        adiak::clustername();
        adiak::value("num_procs", numtasks);
        adiak::value("InputSize", numVals); // The number of elements in input dataset (1000)
        adiak::value("Algorithm", "merge_sort");
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputType", type_of_input);
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        // adiak::value("MPI_Reduce-whole_computation_time", main_time);
        adiak::value("MPI_Reduce-master_initialization_time", data_init);
        adiak::value("MPI_Reduce-master_send_time", comm);
        adiak::value("MPI_Reduce-master_receive_time", comm_large_MPI_Gather);
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("InputSize", numVals); // The number of elements in input dataset (1000)
        adiak::value("InputType", type_of_input); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_threads", "N/A"); // The number of CUDA or OpenMP threads
        adiak::value("num_blocks", "N/A"); // The number of CUDA blocks 
        adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online/AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    

        mtype = FROM_WORKER;
        MPI_Recv(&comm_MPI_Barrier, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&comp_large, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

        adiak::value("MPI_Reduce-worker_comm_MPI_Barrier", comm_MPI_Barrier);
        adiak::value("MPI_Reduce-worker_comp_large", comp_large);
    }
    else if (taskid == 1)
    { // Print only from the first worker.

        mtype = FROM_WORKER;
        MPI_Send(&comm_MPI_Barrier, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&comp_large, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }

    if (taskid == 0)
    {

        int is_correct = check_sorted(values, numVals);
        if (is_correct)
        {
            printf("The array is correctly sorted.\n");
        }
        else
        {
            printf("Error: The array is not correctly sorted.\n");
        }

        free(values);
    }
    
    // CALI_MARK_END(main_time);
    mgr.stop();
    mgr.flush();

    // Finalize MPI
    free(localArray);
    MPI_Finalize();
    return 0;
}

void merge(const float *leftArray, int leftSize, const float *rightArray, int rightSize, float *mergedArray)
{
    int i = 0, j = 0, k = 0;

    while (i < leftSize && j < rightSize)
    {
        if (leftArray[i] < rightArray[j])
        {
            mergedArray[k] = leftArray[i];
            i++;
        }
        else
        {
            mergedArray[k] = rightArray[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArray, if there are any
    while (i < leftSize)
    {
        mergedArray[k] = leftArray[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightArray, if there are any
    while (j < rightSize)
    {
        mergedArray[k] = rightArray[j];
        j++;
        k++;
    }
}

int compare_floats(const void *a, const void *b)
{
    float arg1 = *(const float *)a;
    float arg2 = *(const float *)b;
    if (arg1 < arg2)
        return -1;
    if (arg1 > arg2)
        return 1;
    return 0;
}

float *mergeSortRecursive(int treeDepth, int processId, float *subArray, int subArraySize, MPI_Comm comm, float *fullArray)
{
    int parentId, rightChildId, currentDepth;
    float *sortedSubArray, *receivedArray, *mergedArray;

    currentDepth = 0;
    // Initial local sort of the provided sub-array
    qsort(subArray, subArraySize, sizeof(float), compare_floats);
    sortedSubArray = subArray;

    while (currentDepth < treeDepth)
    {
        parentId = processId & (~(1 << currentDepth));

        if (parentId == processId)
        { // This is a left child or the root
            rightChildId = processId | (1 << currentDepth);

            receivedArray = (float *)malloc(subArraySize * sizeof(float));
            MPI_Recv(receivedArray, subArraySize, MPI_FLOAT, rightChildId, 0, comm, MPI_STATUS_IGNORE);

            mergedArray = (float *)malloc(subArraySize * 2 * sizeof(float));

            if (!mergedArray)
            {
                fprintf(stderr, "Memory allocation failed for merged array.\n");
                exit(EXIT_FAILURE);
            }
            merge(sortedSubArray, subArraySize, receivedArray, subArraySize, mergedArray);

            free(receivedArray);
            // if (currentDepth > 0)
            // {
            //     free(sortedSubArray);
            // }
            sortedSubArray = mergedArray;
            subArraySize *= 2;
            mergedArray = NULL;

            currentDepth++;
        }
        else
        {

            MPI_Send(sortedSubArray, subArraySize, MPI_FLOAT, parentId, 0, comm);
            if (currentDepth > 0)
            {
                free(sortedSubArray);
            }
            break;
        }
    }

    // If this is the root process, copy the sorted array to the fullArray
    if (processId == 0 && fullArray != NULL)
    {
        memcpy(fullArray, sortedSubArray, subArraySize * sizeof(float));
    }
    return fullArray;
}