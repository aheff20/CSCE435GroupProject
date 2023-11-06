cmake_minimum_required(VERSION 3.12)

project(sample_cuda LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../Utils)

set(SOURCE_FILES
    sample_cuda.cu
    ${CMAKE_SOURCE_DIR}/../../Utils/helper_functions.cpp 
)

add_executable(sample_cuda ${SOURCE_FILES})

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(sample_cuda PRIVATE MPI::MPI_CXX)
target_link_libraries(sample_cuda PRIVATE caliper)