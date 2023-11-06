#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

void merge(const float* leftArray, int leftSize, const float* rightArray, int rightSize, float* mergedArray) {
    int i = 0, j = 0, k = 0;

    while (i < leftSize && j < rightSize) {
        if (leftArray[i] < rightArray[j]) {
            mergedArray[k] = leftArray[i];
            i++;
        } else {
            mergedArray[k] = rightArray[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArray, if there are any
    while (i < leftSize) {
        mergedArray[k] = leftArray[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightArray, if there are any
    while (j < rightSize) {
        mergedArray[k] = rightArray[j];
        j++;
        k++;
    }
}

int compare_floats(const void *a, const void *b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

float* mergeSortRecursive(int treeDepth, int processId, float* subArray, int subArraySize, MPI_Comm comm, float* fullArray){
    int parentId, rightChildId, currentDepth;
    float *sortedSubArray, *receivedArray, *mergedArray;

    currentDepth = 0;
    // Initial local sort of the provided sub-array
    qsort(subArray, subArraySize, sizeof(float), compare_floats);
    sortedSubArray = subArray;
    
    while (currentDepth < treeDepth) {
        parentId = processId & (~(1 << currentDepth));

        if (parentId == processId) { // This is a left child or the root
            rightChildId = processId | (1 << currentDepth);

            
            receivedArray = (float*) malloc (subArraySize * sizeof(float));
            MPI_Recv(receivedArray, subArraySize, MPI_FLOAT, rightChildId, 0, comm, MPI_STATUS_IGNORE);

            
            mergedArray = (float*) malloc (subArraySize * 2 * sizeof(float));

            if (!mergedArray) {
                fprintf(stderr, "Memory allocation failed for merged array.\n");
                exit(EXIT_FAILURE);
            }
            merge(sortedSubArray, subArraySize, receivedArray, subArraySize, mergedArray);

          
            free(receivedArray);
            if (currentDepth > 0) {
                free(sortedSubArray);
            }
            sortedSubArray = mergedArray;
            subArraySize *= 2;  

            currentDepth++;
        } else { 
         
            MPI_Send(sortedSubArray, subArraySize, MPI_FLOAT, parentId, 0, comm);
            if (currentDepth > 0) {
                free(sortedSubArray);
            }
            break; 
        }
    }

    // If this is the root process, copy the sorted array to the fullArray
    if(processId == 0 && fullArray != NULL){
        memcpy(fullArray, sortedSubArray, subArraySize * sizeof(float));
    }
    return fullArray; 
}





int main(int argc, char *argv[])
{

        int numVals;
    if (argc == 2)
    {
        numVals = atoi(argv[1]);
    }
    else
    {
        fprintf(stderr, "\nUsage: %s <number_of_values>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype;                 /* message type */
    float *values = (float*)malloc(numVals * sizeof(float));
    double whole_computation_time,master_initialization_time = 0;
    int height = 0;
    

    MPI_Status status;
     // Define Caliper region names
    const char* main = "main";
    const char* comm = "comm";
    const char* comm_MPI_Barrier = "comm_MPI_Barrier";
    const char* comm_large = "comm_large";
    const char* comm_large_MPI_Gather = "comm_large_MPI_Gather";
    const char* comm_large_MPI_Scatter = "comm_large_MPI_Scatter";
    const char* comp = "comp";
    const char* comp_large = "comp_large";
    const char* data_init = "data_init";
    CALI_CXX_MARK_FUNCTION;

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
    numworkers = numtasks-1;
    height = log2(numtasks);

    CALI_MARK_BEGIN(main);
    double start3, end3, startTime,localTime,totalTime= 0;
    start3 =MPI_Wtime();

     // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialization
    if (taskid == MASTER)
    {

        printf("merge_mpi has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");
        double start, end;
        start = MPI_Wtime();
        CALI_MARK_BEGIN(data_init);
        array_fill_random(values, numVals);
        CALI_MARK_END(data_init);
        end =MPI_Wtime();
        master_initialization_time = end -start;

        

        // CALI_MARK_BEGIN(comm_large_MPI_Gather);
        // // Collect the sorted segments from the worker tasks
        // // ...
        // // Here you would potentially use MPI_Gather which is timed here
        // CALI_MARK_END(comm_large_MPI_Gather);
    }

    int localArraySize = numVals / numtasks;
    float *localArray = (float*) malloc(localArraySize * sizeof(float));
    CALI_MARK_BEGIN(comm); 
    MPI_Scatter(values, localArraySize, MPI_FLOAT, localArray, localArraySize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm);

     //Merge sort
   if (taskid == 0) {
        double zeroStartTime = MPI_Wtime();
        // Assuming mergeSort returns a pointer to the sorted array, which is unusual.
        float *sortedGlobalArray = mergeSortRecursive(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, values);
        double zeroTotalTime = MPI_Wtime() - zeroStartTime;
       

        int is_correct = check_sorted(sortedGlobalArray, numVals);
        if (is_correct) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("Error: The array is not correctly sorted.\n");
        }

       
        free(localArray);
        if (sortedGlobalArray != values) {
            free(sortedGlobalArray);
        }
    }

    else {
            double processStartTime = MPI_Wtime();
        // As a worker, you do not need to manage the global array.
        mergeSortRecursive(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, NULL);
        double processTotalTime = MPI_Wtime() - processStartTime;
       
        
        free(localArray);
    }



    // if (taskid > MASTER)
    // {
    //     // CALI_MARK_BEGIN(comm_MPI_Barrier);
    //     // // Workers wait for the initial data
    //     // // Here you might have an MPI_Barrier, if so, it is timed here
    //     // CALI_MARK_END(comm_MPI_Barrier);

    //     CALI_MARK_BEGIN(comp_large);
    //     // Receive the segment of the array, sort it, and then send it back
    //     // Here you would potentially use MPI_Recv and MPI_Send which are parts of the "comp_large" and "comm_large" respectively
    //     CALI_MARK_END(comp_large);
    // }

    // Finalize the main computation region
    CALI_MARK_END(main);
    end3 = MPI_Wtime();
    whole_computation_time = end3 -start3;

    localTime = MPI_Wtime() - startTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);

   adiak::init(NULL);
   adiak::user();
   adiak::launchdate();
   adiak::libraries();
   adiak::cmdline();
   adiak::clustername();
   adiak::value("num_procs", numtasks);
   adiak::value("num_vals", numVals);
   adiak::value("program_name", "merge_sort_mpi");
   adiak::value("array_datatype_size", sizeof(double));



   

    MPI_Comm worker_comm;
   MPI_Comm_split(MPI_COMM_WORLD, 1, 1, &worker_comm);



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
      adiak::value("MPI_Reduce-whole_computation_time", main);
      adiak::value("MPI_Reduce-master_initialization_time", data_init);
      adiak::value("MPI_Reduce-master_send_time", comm);
      adiak::value("MPI_Reduce-master_receive_time", comm_large_MPI_Gather);


      mtype = FROM_WORKER;
      MPI_Recv(&comm_MPI_Barrier, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&comp_large, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);


      adiak::value("MPI_Reduce-worker_comm_MPI_Barrier", comm_MPI_Barrier);
      adiak::value("MPI_Reduce-worker_comp_large", comp_large);


   }
   else if (taskid == 1)
   { // Print only from the first worker.
      // Print out worker time results.
      
      // Compute averages after MPI_Reduce
    //   worker_recieve_time_average = worker_receive_time_sum / (double)numworkers;
    //   worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
    //   worker_send_time_average = worker_send_time_sum / (double)numworkers;

    //   printf("******************************************************\n");
    //   printf("Worker Times:\n");
    //   printf("Worker wait for data time: %f \n", comm_MPI_Barrier);
    //   printf("Worker comp time: %f \n", comp_large);
    //   printf("\n******************************************************\n");

      mtype = FROM_WORKER;
      MPI_Send(&comm_MPI_Barrier, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&comp_large, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
     
   }

    mgr.stop();
    mgr.flush();

    // Finalize MPI
    free(values);
    MPI_Finalize();
    return 0;
}
