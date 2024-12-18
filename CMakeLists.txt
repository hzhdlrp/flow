cmake_minimum_required(VERSION 3.26)
project(untitled3)

add_executable(untitled_main4 main4.cpp)

#add_executable(untitled3_main2 main2.cpp)
#add_executable(untitled33_main main.cpp)
#add_executable(untitled_main3 main3.cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++20")
find_package(OpenMp REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(untitled_main4 PRIVATE OpenMP::OpenMP_CXX)
endif ()



