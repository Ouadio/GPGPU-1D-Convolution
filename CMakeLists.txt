cmake_minimum_required(VERSION 3.10)

project(conv1D VERSION 1.0  LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)

SET(CUDA_INCLUDE_DIRS "/opt/cuda/samples/common/inc" ${CUDA_INCLUDE_DIRS})

add_subdirectory(src conv1DLibrary)

add_executable(conv1DLauncher main.cpp)

# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_include_directories(conv1DLauncher PUBLIC "${CUDA_INCLUDE_DIRS}" "${PROJECT_BINARY_DIR}")

target_link_libraries(conv1DLauncher conv1DLibrary)

#################### SEPERATE COMPILATION AND LINKING OF C++ DEVICE CODE #####################
#'Using object linking of device code, the compiler can generate device code for all functions 
# in a .cpp file, store it in a .o file, and then link device code from multiple .o files together 
# in the same way that we are used to linking CPU code. As a result, the build structure does not 
# change much. 