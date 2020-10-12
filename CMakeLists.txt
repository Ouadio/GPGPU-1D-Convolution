cmake_minimum_required(VERSION 3.10)

project(conv1D VERSION 1.0  LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)

SET(CUDA_INCLUDE_DIRS "/opt/cuda/samples/common/inc" ${CUDA_INCLUDE_DIRS})

add_subdirectory(conv1DLibrary conv1DLibrary)

add_executable(conv1DLauncher main.cu) 

target_include_directories(conv1DLauncher PUBLIC "${CUDA_INCLUDE_DIRS}" "${PROJECT_BINARY_DIR}")

target_link_libraries(conv1DLauncher conv1DLibrary)

