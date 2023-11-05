cmake_minimum_required(VERSION 3.12)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/../Utils)

set(SOURCE_FILES
    merge_mpi.cpp
    ${CMAKE_SOURCE_DIR}/../../Utils/helper_functions.cpp 
)

add_executable(merge_mpi ${SOURCE_FILES})


message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(merge_mpi PRIVATE MPI::MPI_CXX)
target_link_libraries(merge_mpi PRIVATE caliper)
