cmake_minimum_required(VERSION 3.16)
project(Lab1 C)
set(CMAKE_C_STANDARD 99)

message(STATUS "OpenCL CMake loading...")

set(SOURCE_FILES main.c)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

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