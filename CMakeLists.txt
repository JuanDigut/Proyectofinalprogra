cmake_minimum_required(VERSION 3.16)
project(TensorFlowCppExamples VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable export of compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find TensorFlow C++ library
# The user needs to set TENSORFLOW_DIR to the TensorFlow installation directory
# or install TensorFlow system-wide
if(DEFINED ENV{TENSORFLOW_DIR})
    set(TENSORFLOW_DIR $ENV{TENSORFLOW_DIR})
endif()

if(TENSORFLOW_DIR)
    message(STATUS "Using TensorFlow from: ${TENSORFLOW_DIR}")
    include_directories(${TENSORFLOW_DIR}/include)
    link_directories(${TENSORFLOW_DIR}/lib)
else()
    # Try to find TensorFlow using pkg-config
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(TENSORFLOW tensorflow)
    endif()
    
    if(NOT TENSORFLOW_FOUND)
        message(WARNING "TensorFlow not found. Please set TENSORFLOW_DIR environment variable or install TensorFlow.")
        message(STATUS "Example: cmake -DTENSORFLOW_DIR=/path/to/tensorflow ..")
    endif()
endif()

# Add examples
add_subdirectory(examples)

# Print configuration summary
message(STATUS "")
message(STATUS "=== TensorFlow C++ Examples Configuration ===")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
