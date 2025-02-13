cmake_minimum_required(VERSION 3.30.0)
project(CannyEdge)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

add_subdirectory(src)

# Test directory
enable_testing()
add_subdirectory(tests)