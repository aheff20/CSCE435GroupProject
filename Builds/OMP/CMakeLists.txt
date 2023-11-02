cmake_minimum_required(VERSION 3.12)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "-O -fopenmp")

add_executable(quicksort quicksort.cpp)
target_link_libraries(quicksort PRIVATE caliper)

message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})