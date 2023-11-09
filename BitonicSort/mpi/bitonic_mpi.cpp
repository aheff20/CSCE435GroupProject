#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

void high_bit(int bit);
void low_bit(int bit);
int compare_floats(const void *a, const void *b);

#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

int numtasks,   /* number of tasks in partition */
    taskid,     /* a task identifier */
    numworkers, /* number of worker tasks */
    source,     /* task id of message source */
    dest,       /* task id of message destination */
    mtype;      /* message type */

double whole_computation_time, master_initialization_time = 0;

float *local_array;
float *global_array;
int array_size;

MPI_Status status;
// Define Caliper region names
const char *main = "main";
const char *comm = "comm";
const char *comm_MPI_Barrier = "comm_MPI_Barrier";
const char *comm_large = "comm_large";
const char *comm_large_MPI_Gather = "comm_large_MPI_Gather";
const char *comm_large_MPI_Scatter = "comm_large_MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *data_init = "data_init";

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN(main);

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

    // Initialize MPI and determine rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    float *values = (float *)malloc(numVals * sizeof(float));

    if (numtasks < 2)
    {
        fprintf(stderr, "Need at least two MPI tasks. Quitting...\n");
        MPI_Finalize();
        free(values);
        return EXIT_FAILURE;
    }

    // data initalizationn

    array_size = numVals / numtasks;

    local_array = (float *)malloc(array_size * sizeof(float));
    if (taskid == MASTER)
    {
        global_array = (float *)malloc(global_size * sizeof(float));
    }
    CALI_MARK_BEGIN(data_init);
    array_fill_random(local_array, array_size);
    CALI_MARK_END(data_init);
    // MP baried
    MPI_Barrier(MPI_COMM_WORLD);
    int proc_step = (int)(log2(numtasks));
    // local sort  in owrker
    qsort(local_array, array_size, sizeof(float), compare_floats);

    // iteratue over stages, porcesses, and cal high or low
    for (int i = 0; i < proc_step; i++)
    {
        for (int j = i; j >= 0; j--)
        {
            if (((taskid >> (i + 1)) % 2 == 0 && (taskid >> j) % 2 == 0) || ((taskid >> (i + 1)) % 2 != 0 && (taskid >> j) % 2 != 0))
            {
                Low(j);
            }
            else
            {
                High(j);
            }
        }
    }

    // mpi GATHER for local to global::
    MPI_Gather(local_array, array_size, MPI_FLOAT, global_array, array_size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER)
    {

        int is_correct = check_sorted(global_array, numVals);
        if (is_correct)
        {
            printf("The array is correctly sorted.\n");
        }
        else
        {
            printf("Error: The array is not correctly sorted.\n");
        }
    }

    CALI_MARK_END(main);

    free(local_array);
    if (taskid == MASTER)
    {
        free(global_array);
    }
    MPI_Finalize();

    return 0;
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

void high_bit(int stage_bit)
{
    int i;
    float partner_max;

    // Allocate buffers for send and receive
    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int receive_count;
    MPI_Recv(&partner_max, 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Prepare send buffer
    int send_count = 0;
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    MPI_Send(&local_array[0], 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    // Populate send buffer with values less than partner's max
    for (i = 0; i < array_size; i++)
    {
        if (local_array[i] < partner_max)
        {
            send_buffer[send_count + 1] = local_array[i];
            send_count++;
        }
        else
        {
            break;
        }
    }

    // Exchange data with partner process
    MPI_Recv(receive_buffer, array_size, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    receive_count = (int)receive_buffer[0];
    send_buffer[0] = (float)send_count;
    MPI_Send(send_buffer, send_count + 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    // Merge received values
    for (i = 1; i <= receive_count; i++)
    {
        if (receive_buffer[i] > local_array[0])
        {
            local_array[0] = receive_buffer[i];
        }
        else
        {
            break;
        }
    }

    // Sort the updated local array
    qsort(local_array, array_size, sizeof(float), Comparison);
    free(send_buffer);
    free(receive_buffer);
}

void low_bit(int stage_bit)
{
    int i;
    float partner_min;

    // Allocate buffers for send and receive
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int send_count = 0;
    MPI_Send(&local_array[array_size - 1], 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    int receive_count;
    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    MPI_Recv(&partner_min, 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Populate send buffer with values greater than partner's min
    for (i = array_size - 1; i >= 0; i--)
    {
        if (local_array[i] > partner_min)
        {
            send_buffer[send_count + 1] = local_array[i];
            send_count++;
        }
        else
        {
            break;
        }
    }

    send_buffer[0] = (float)send_count;
    MPI_Send(send_buffer, send_count + 1, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD);
    MPI_Recv(receive_buffer, array_size, MPI_FLOAT, process_rank ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Merge received values
    receive_count = (int)receive_buffer[0];
    for (i = 1; i <= receive_count; i++)
    {
        if (local_array[array_size - 1] < receive_buffer[i])
        {
            local_array[array_size - 1] = receive_buffer[i];
        }
        else
        {
            break;
        }
    }

    // Sort the updated local array
    qsort(local_array, array_size, sizeof(float), Comparison);
    free(send_buffer);
    free(receive_buffer);
};