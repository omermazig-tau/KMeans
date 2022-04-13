cmake_minimum_required(VERSION 3.20)
project(CPart C)

set(CMAKE_C_STANDARD 90)

add_executable(CPart
        kmeans.c spkmeans.c)
