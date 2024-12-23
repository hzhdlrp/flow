cmake_minimum_required(VERSION 3.26)
project(untitled3)

add_executable(proj main.cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++20")
find_package(OpenMp REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(proj PRIVATE OpenMP::OpenMP_CXX)
endif ()



