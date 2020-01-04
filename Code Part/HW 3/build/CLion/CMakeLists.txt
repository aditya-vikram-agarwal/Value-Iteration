#cmake_minimum_required(VERSION 3.13)
project(CLion)

# Set requirements and C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the executable we want to build
add_executable(CLion ../../main.cpp)

# Set include directory
target_include_directories(CLion PUBLIC ../../lib)
