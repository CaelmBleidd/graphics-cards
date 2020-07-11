cmake_minimum_required(VERSION 3.16)
project(Lab1 C)
set(CMAKE_C_STANDARD 99)

message(STATUS "OpenCL CMake loading...")

set(SOURCE_FILES main.c)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -lm")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})


message(STATUS "Checking OpenCL...")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

if (OPENCL_FOUND)
    message(STATUS "Found OpenCL!")
    # add OpenCL to project
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
else ()
    message(STATUS "Missed OpenCL!")
endif ()


find_package(OpenMP)

if (OPENMP_FOUND)
    message(STATUS "Found OpenMP!")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(STATUS "Missed OpenMP!")
endif ()