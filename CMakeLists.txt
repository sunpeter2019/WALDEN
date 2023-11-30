project(walden)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 14)

# Define the macro ‘DEBUG' in the debug mode
if(CMAKE_BUILD_TYPE STREQUAL Debug)        
    ADD_DEFINITIONS(-DDEBUG)               
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /W1 /EHsc")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "-O3 -xHost")
else()
    # clang and gcc
    set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(src/core)

add_executable(benchmark src/examples/benchmark/main.cpp)
add_executable(shift src/examples/benchmark/shift.cpp)
add_executable(shift-thread src/examples/benchmark/shift-thread.cpp)
add_executable(example src/examples/example.cpp)
add_executable(example_bulkload src/examples/example_bulkload.cpp)