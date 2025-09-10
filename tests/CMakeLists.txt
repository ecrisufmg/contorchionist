cmake_minimum_required(VERSION 3.15)

# Include the cpp tests subdirectory - this will define all the C++ test targets
add_subdirectory(core_tests/cpp)

# NOTE: Python tests are in core_tests/python/ and don't require CMake build
