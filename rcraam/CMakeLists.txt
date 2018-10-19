cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

project(RCRAAM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

include_directories(../)
include_directories(~/R/x86_64-pc-linux-gnu-library/3.5/Rcpp/include/)

add_executable(${PROJECT_NAME}
    "src/utils.hpp"
    "src/robust_algorithms.cpp")
