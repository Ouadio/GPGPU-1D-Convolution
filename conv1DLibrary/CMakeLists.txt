add_library(conv1DLibrary conv1DKernel.cu)

target_include_directories(conv1DLibrary PUBLIC "${PROJECT_BINARY_DIR}" "${CUDA_INCLUDE_DIRS}" )

target_include_directories(conv1DLibrary INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

