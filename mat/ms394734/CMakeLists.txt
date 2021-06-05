cmake_minimum_required(VERSION 3.5)
project (MIMUW-matgen)

SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14  -g3 -Wall -O0")

add_executable(generator
        densematgen.cpp
        densematgen.h
        generator_main.cpp)

add_executable(matrixmul
        densematgen.cpp
        densematgen.h
        utils.cpp
        utils.h
        colA.cpp
        colA.h
        innerABC.cpp
        innerABC.h
        matrixmul.cpp)

# add_executable(generator_all
#         densematgen.cpp
#         densematgen.h
#         generator_all.cpp)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(matrixmul PUBLIC OpenMP::OpenMP_CXX)
endif()

# MPI
find_package(MPI REQUIRED)
target_link_libraries(matrixmul PUBLIC MPI::MPI_CXX)
