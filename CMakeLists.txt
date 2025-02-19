cmake_minimum_required(VERSION 3.28.0)
project(CannyEdge)

option(ENABLE_CUDA ON)
if(ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
    else()
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

add_subdirectory(src)

add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Test directory
enable_testing()
add_subdirectory(tests)
