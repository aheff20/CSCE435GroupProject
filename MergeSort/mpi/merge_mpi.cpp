#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


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
    int taskid, numtasks;
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

    CALI_MARK_BEGIN(main);
    double start3, end3 = 0;
    start3 =MPI_Wtime();

     // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialization
    if (taskid == MASTER)
    {

        printf("merge_mpi has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");
        CALI_MARK_BEGIN(data_init);
        array_fill_random(values, numVals);
        CALI_MARK_END(data_init);

        CALI_MARK_BEGIN(comm);
        // Send parts of the array to worker tasks
        // ...
        // Here you would potentially use MPI_Scatter which is a part of "comm_large_MPI_Scatter"
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comm_large_MPI_Gather);
        // Collect the sorted segments from the worker tasks
        // ...
        // Here you would potentially use MPI_Gather which is timed here
        CALI_MARK_END(comm_large_MPI_Gather);
    }


    if (taskid > MASTER)
    {
        CALI_MARK_BEGIN(comm_MPI_Barrier);
        // Workers wait for the initial data
        // Here you might have an MPI_Barrier, if so, it is timed here
        CALI_MARK_END(comm_MPI_Barrier);

        CALI_MARK_BEGIN(comp_large);
        // Receive the segment of the array, sort it, and then send it back
        // Here you would potentially use MPI_Recv and MPI_Send which are parts of the "comp_large" and "comm_large" respectively
        CALI_MARK_END(comp_large);
    }

    // Finalize the main computation region
    CALI_MARK_END(main);
    end3 = MPI_Wtime();
    whole_computation_time = end3 -start3;

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



    double comm_MPI_Barrier,
      comp_large = 0; // Worker statistic values.

      MPI_Comm worker_comm;
   MPI_Comm_split(MPI_COMM_WORLD, 1, 1, &worker_comm);


    if (taskid == 0)
   {
    // Master Times
      printf("******************************************************\n");
      printf("Master Times:\n");
      printf("Whole Computation Time: %f \n", main);
      printf("Master Initialization Time: %f \n", data_init);
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

      printf("******************************************************\n");
      printf("Worker Times:\n");
      printf("Worker wait for data time: %f \n", comm_MPI_Barrier);
      printf("Worker comp time: %f \n", comp_large);
      printf("\n******************************************************\n");

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