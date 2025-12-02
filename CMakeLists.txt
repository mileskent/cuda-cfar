# Minimum required version for modern CMake and CUDA support
cmake_minimum_required(VERSION 3.18)

# Define your project name
project(
    CudaMatplotExample
    LANGUAGES CXX CUDA # Enable C++ and CUDA language support
)

# --- Configuration ---

# Set C++ standard (Matplot++ often requires C++17 or newer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# --- Matplot++ Integration ---

# Add the Matplot++ source directory. 
# NOTE: This assumes the matplotplusplus repository is in a subdirectory 
# named 'matplotplusplus' relative to this CMakeLists.txt.
add_subdirectory(matplotplusplus)

# --- Executable Definition ---

# Create the executable target. 
# We use the correct source file name: main.cu
add_executable(
    main
    main.cu
)

# --- Link Libraries ---

# Link the executable against the required libraries:
# 1. matplot: The main Matplot++ library.
# 2. CUDA::cudart: The CUDA Runtime Library, required when using the CUDAToolkit.

target_link_libraries(
    main
    PUBLIC 
    matplot 
    cufft
)