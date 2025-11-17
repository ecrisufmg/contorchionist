cmake_minimum_required(VERSION 3.18)
project(PythonWrapper CXX)

message(STATUS "Python Wrapper CMakeLists.txt (contorchionist project):")

# --- Pybind11 ---
# find_package(pybind11 REQUIRED)
# add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11)
message(STATUS "  pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "  pybind11 library: ${pybind11_LIBRARIES}") # Modern pybind11 uses targets

# --- Source Files ---
# Use the new organized structure
set(PYTHON_MODULE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/pycontorchionist/pycontorchionist.cpp)

if(NOT EXISTS ${PYTHON_MODULE_SRC})
    message(WARNING "Python module source file not found: ${PYTHON_MODULE_SRC}. Please create it.")

    # Create a dummy file to allow CMake to configure, replace with actual binding code
    file(WRITE ${PYTHON_MODULE_SRC} "// Dummy file for ${PYTHON_MODULE_SRC}\n// Please replace with your pybind11 binding code.\n#include <pybind11/pybind11.h>\nPYBIND11_MODULE(pycontorchionist, m) { m.doc() = \"pybind11 contorchionist plugin\"; }")
    message(STATUS "Created dummy source file: ${PYTHON_MODULE_SRC}")
endif()

# --- Target Definition: Python Module ---
set(PYTHON_MODULE_NAME "pycontorchionist") # Name of the .so/.pyd file

message(STATUS "Configuring Python module: ${PYTHON_MODULE_NAME}")
message(STATUS "  Source: ${PYTHON_MODULE_SRC}")

pybind11_add_module(${PYTHON_MODULE_NAME} MODULE ${PYTHON_MODULE_SRC})

# --- Include Directories ---
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE
    ${pybind11_INCLUDE_DIRS} # Added by pybind11_add_module typically

    # Public headers of the main project (contorchionist_core)
    ${CMAKE_SOURCE_DIR}/include
)

# --- Linking ---
if(TARGET contorchionist::core)
    target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE contorchionist::core)
    message(STATUS "Linking ${PYTHON_MODULE_NAME} with contorchionist::core")
elseif(TARGET contorchionist_core)
    target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE contorchionist_core)
    message(STATUS "Linking ${PYTHON_MODULE_NAME} with contorchionist_core")
else()
    message(WARNING "Target 'contorchionist_core' (or alias contorchionist::core) not found. Cannot link ${PYTHON_MODULE_NAME}.")
endif()

# Link with pybind11 libraries (modern pybind11 uses targets like pybind11::module or pybind11::embed)
# pybind11_add_module should handle this, but explicitly linking can sometimes be necessary
# target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE pybind11::module) # Or pybind11::embed if appropriate

# --- Output Name & Directory ---
set(PYTHON_WRAPPER_OUTPUT_DIR "${PROJECT_DIST_DIR}/python")
set(PYTHON_PACKAGE_DIR "${PYTHON_WRAPPER_OUTPUT_DIR}/pycontorchionist")
file(MAKE_DIRECTORY ${PYTHON_WRAPPER_OUTPUT_DIR})
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DIR})

# Set output directory for the compiled module to go inside the package
set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}"
)

message(STATUS "Output directory for ${PYTHON_MODULE_NAME} set to: ${PYTHON_PACKAGE_DIR}")
message(STATUS "Python package directory: ${PYTHON_PACKAGE_DIR}")

# --- RPATH settings ---
# For Python modules, rely on the user's PyTorch installation
# No explicit RPATH needed as we don't distribute LibTorch
if(APPLE)
    set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX AND NOT APPLE)
    set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# --- NOTE: Copy Core Library but NOT LibTorch ---
# This wrapper needs libcontorchionist_core.dylib but relies on the user's 
# installed PyTorch for LibTorch libraries to avoid version conflicts.

# Copy contorchionist_core library to the Python package directory
add_custom_command(
    TARGET ${PYTHON_MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:contorchionist_core>
    ${PYTHON_PACKAGE_DIR}
    COMMENT "Copying contorchionist_core shared library to ${PYTHON_PACKAGE_DIR}"
)

# --- Copy Example Files ---
# Examples should be copied directly to the distribution root
set(PYTHON_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
set(PYTHON_DIST_ROOT_DIR "${CMAKE_BINARY_DIR}/dist/python")
if(EXISTS ${PYTHON_EXAMPLES_DIR})
    # Copy each file in examples directory to the distribution root
    file(GLOB EXAMPLE_FILES "${PYTHON_EXAMPLES_DIR}/*")
    foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
        get_filename_component(FILENAME ${EXAMPLE_FILE} NAME)
        add_custom_command(
            OUTPUT ${PYTHON_DIST_ROOT_DIR}/${FILENAME}
            COMMAND ${CMAKE_COMMAND} -E copy ${EXAMPLE_FILE} ${PYTHON_DIST_ROOT_DIR}/${FILENAME}
            DEPENDS ${PYTHON_MODULE_NAME} ${EXAMPLE_FILE}
            COMMENT "Copying example file ${FILENAME} to Python distribution root"
        )
        list(APPEND EXAMPLE_OUTPUTS ${PYTHON_DIST_ROOT_DIR}/${FILENAME})
    endforeach()
    add_custom_target(python_copy_examples ALL DEPENDS ${EXAMPLE_OUTPUTS})
    message(STATUS "Configured copying of example files from ${PYTHON_EXAMPLES_DIR} to ${PYTHON_DIST_ROOT_DIR}")
endif()

# --- Copy Python Package Configuration Files ---
set(PYTHON_CONFIG_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
)

foreach(CONFIG_FILE ${PYTHON_CONFIG_FILES})
    if(EXISTS ${CONFIG_FILE})
        add_custom_command(
            TARGET ${PYTHON_MODULE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CONFIG_FILE} ${PYTHON_WRAPPER_OUTPUT_DIR}
            COMMENT "Copying ${CONFIG_FILE} to ${PYTHON_WRAPPER_OUTPUT_DIR}"
        )
    endif()
endforeach()

# --- Create Python Package __init__.py ---
set(INIT_PY_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/src/pycontorchionist/__init__.py")
if(EXISTS ${INIT_PY_TEMPLATE})
    add_custom_command(
        TARGET ${PYTHON_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${INIT_PY_TEMPLATE} ${PYTHON_PACKAGE_DIR}/__init__.py
        COMMENT "Copying Python package __init__.py from template"
    )
    message(STATUS "Will copy __init__.py from ${INIT_PY_TEMPLATE}")
else()
    # Fallback to generated __init__.py if template doesn't exist
    file(WRITE ${CMAKE_BINARY_DIR}/init_py_content.txt 
"\"\"\"
pycontorchionist - Python bindings for Contorchionist Audio Processing Library

This package provides high-performance audio processing capabilities with LibTorch backend.
\"\"\"

# Import torch first to load LibTorch libraries into memory
# This is required for the compiled extension to find the shared libraries
import torch
import torchaudio

from .pycontorchionist import *

__version__ = \"0.1.0\"
__author__ = \"Contorchionist Team\"

# Re-export main classes and functions for convenience
from .pycontorchionist import (
    MelSpectrogramProcessor,
    MelNormMode,
    SpectrumDataFormat, 
    MelFormulaType,
    mel_norm_mode_to_string,
    string_to_mel_norm_mode,
    spectrum_data_format_to_string,
    string_to_spectrum_data_format,
)

__all__ = [
    \"MelSpectrogramProcessor\",
    \"MelNormMode\",
    \"SpectrumDataFormat\",
    \"MelFormulaType\", 
    \"mel_norm_mode_to_string\",
    \"string_to_mel_norm_mode\",
    \"spectrum_data_format_to_string\",
    \"string_to_spectrum_data_format\",
]
")

    add_custom_command(
        TARGET ${PYTHON_MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/init_py_content.txt ${PYTHON_PACKAGE_DIR}/__init__.py
        COMMENT "Creating Python package __init__.py from fallback"
    )
endif()