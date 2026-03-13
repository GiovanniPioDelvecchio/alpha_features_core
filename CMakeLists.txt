cmake_minimum_required(VERSION 3.14)
project(alpha_features_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(alpha_features_src)
include_directories(${CMAKE_SOURCE_DIR})