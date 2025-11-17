cmake_minimum_required(VERSION 3.15)
project(AudioMultiSys LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Options to control which parts of the project are built
option(BUILD_PD_INTEGRATION "Build the Pure Data integration" ON)
option(BUILD_MAX_INTEGRATION "Build the Max/MSP integration" OFF)
option(BUILD_SC_INTEGRATION "Build the SuperCollider integration" OFF)
option(BUILD_PYTHON_INTEGRATION "Build the Python integration" OFF)

add_subdirectory(shared_core)

if(BUILD_PD_INTEGRATION)
    add_subdirectory(wrappers/pd)
endif()

if(BUILD_MAX_INTEGRATION)
    add_subdirectory(wrappers/max)
endif()

if(BUILD_SC_INTEGRATION)
    add_subdirectory(wrappers/sc)
endif()

if(BUILD_PYTHON_INTEGRATION)
    add_subdirectory(wrappers/python)
endif()

# TODO: Add other integrations (SuperCollider, Python) when ready
