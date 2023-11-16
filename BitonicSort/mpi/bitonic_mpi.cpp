#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../../Utils/helper_functions.h"

int totalValues;
int sortType;
int totalProcesses, processId;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";

const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* mpi_gather_region = "MPI_Gather";

const char* mpi_bcast_region = "MPI_Bcast";
const char* correctness_check = "correctness_check";

const char* sortOptions[3] = {"random", "sorted", "reverse_sorted"};

float *createArray(int size, int start, int type) {
    float *data = (float *)malloc(sizeof(float) * size);
    switch (type) {
        case 0:
            array_fill_random_no_seed(data, size);
            break;
        case 1:
            array_fill_descending_local(data, size, processId, totalValues);
            break;
        case 2:
            array_fill_ascending_local(data, size, processId);
            break;
        case 3:
            array_fill_ascending_local(data, size, processId);
            perturb_array(data, size, 0.01);
            break;
        default:
            array_fill_random_no_seed(data, size);
            break;
    }
    return data;
}

void print_array(float* array, int size) {
    for (int i = 0; i < size; i++){
        printf("%0.3f,", array[i]);
    }
    printf("\n");
}

int sortDesc(const void *a, const void *b) {
    float first = *((float *)a);
    float second = *((float *)b);
    if (first < second) return 1;
    if (first > second) return -1;
    return 0;
}

int sortAsc(const void *a, const void *b) {
    float first = *((float *)a);
    float second = *((float *)b);
    if (first < second) return -1;
    if (first > second) return 1;
    return 0;
}


float *tempData;

void exchangeAndSort(float *data, int count, int proc1, int proc2, int descOrder,int seqNo) {
  if (proc1 != processId && proc2 != processId) return;

  memcpy(tempData, data, count*sizeof(float));

  MPI_Status status;

  int otherProc = proc1==processId ? proc2 : proc1;
  if (proc1 == processId) {
    MPI_Send(data, count, MPI_FLOAT, otherProc, seqNo, MPI_COMM_WORLD);
    MPI_Recv(&tempData[count], count, MPI_FLOAT, otherProc, seqNo, MPI_COMM_WORLD, &status);
  }
  else {
    MPI_Recv(&tempData[count], count, MPI_FLOAT, otherProc, seqNo, MPI_COMM_WORLD, &status);
    MPI_Send(data, count, MPI_FLOAT, otherProc, seqNo, MPI_COMM_WORLD);
  }

  if (descOrder) {
    qsort(tempData, count*2, sizeof(float), sortDesc);
  }
  else {
    qsort(tempData, count*2, sizeof(float), sortAsc);
  }

  if (proc1 == processId)
    memcpy(data, tempData, count*sizeof(float));
  else
    memcpy(data, &tempData[count], count*sizeof(float));
}

void bitonicMergeSort(float *data, int count) {
  tempData = (float *) malloc(sizeof(float) * count * 2);

  int logN = totalProcesses;
  int powerOf2 = 2;
  int seqNo = 0;

  for(int i=1; logN > 1 ; i++) {
    int powerOf2j = powerOf2;
    for(int j=i; j >= 1; j--) {
      seqNo++;
      for(int proc=0; proc < totalProcesses; proc += powerOf2j) {
	for(int k=0; k < powerOf2j/2; k++) {
	  exchangeAndSort(data, count, proc+k, proc+k+powerOf2j/2, ((proc+k) % (powerOf2*2) >= powerOf2),seqNo);
	}
      }
      powerOf2j /= 2;
    }
    powerOf2 *= 2;
    logN /= 2;
  }

  free(tempData);
}

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;
  int arraySize;
  long int retVal;
  int nameLen;
  char hostName[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &processId);
  MPI_Get_processor_name(hostName, &nameLen);

  cali::ConfigManager configManager;
  configManager.start();

  totalValues = atoi(argv[1]);
  sortType = atoi(argv[2]);
  arraySize = totalValues / totalProcesses;

  int start = processId*arraySize;
  CALI_MARK_BEGIN(data_init);
  float * data = createArray(arraySize, start, sortType);
  CALI_MARK_END(data_init);
  print_array(data, arraySize);

  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  bitonicMergeSort(data, arraySize);
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  float * allData = NULL;
  if (processId == 0) {
    allData = (float *) malloc(arraySize * totalProcesses * sizeof(float));
  }

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(mpi_gather_region);
  MPI_Gather(data, arraySize, MPI_FLOAT, allData, arraySize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  CALI_MARK_END(mpi_gather_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  
  if (processId == 0) {
    CALI_MARK_BEGIN(correctness_check);
    int sortedCorrectly = check_sorted(allData, arraySize * totalProcesses);
    if (sortedCorrectly)
      printf("Successfully sorted!\n");
    else
      printf("Error: data not sorted.\n");
    CALI_MARK_END(correctness_check);
    
    print_array(allData, arraySize * totalProcesses);

    free(allData);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "bitonic_sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "float");
    adiak::value("SizeOfDatatype", sizeof(float));
    adiak::value("InputSize", totalValues);
    adiak::value("InputType", sortOptions[sortType-1]);
    adiak::value("num_procs", totalProcesses);
    adiak::value("group_num", 1);
    adiak::value("implementation_source", "Online");
    adiak::value("correctness", sortedCorrectly);

  }
  
  free(data);

  configManager.stop();
  configManager.flush();

  MPI_Finalize();
  return 0;
}
