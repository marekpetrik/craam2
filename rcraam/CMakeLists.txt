# a fake cmake file to be used as a project for editing the package with QT Creator or like
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

project(RCRAAM)

find_package(OpenMP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)


# use OpenMP if possible and not debugging
if (OPENMP_FOUND AND NOT (CMAKE_BUILD_TYPE STREQUAL "Debug") )
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(../
                    ./inst/include
                    ~/R/x86_64-pc-linux-gnu-library/4.0/Rcpp/include/
                    ~/R/x86_64-pc-linux-gnu-library/4.0/RcppProgress/include
                    /usr/include/R/)

find_path(RCPP "Rcpp.h" PATHS "~/R/x86_64-pc-linux-gnu-library/4.0/Rcpp/include/")


add_executable(${PROJECT_NAME}
    "src/utils.hpp"
    "src/robust_algorithms.cpp"
    "src/simulation.cpp")
