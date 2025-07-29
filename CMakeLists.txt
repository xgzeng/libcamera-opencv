cmake_minimum_required(VERSION 3.28)

project(libcamera-opencv LANGUAGES CXX)

option(BUILD_SAMPLES "build examples" ON)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(libcamera REQUIRED libcamera IMPORTED_TARGET)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(src)

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()