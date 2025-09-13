cmake_minimum_required(VERSION 3.15)

project(contorchionist LANGUAGES CXX C)

# Generate compile_commands.json for IDEs like VSCode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Build Options for Wrappers (can be defined here or in src/wrappers/CMakeLists.txt) ---
option(BUILD_PD_WRAPPER "Build the Pure Data wrapper" OFF)
option(BUILD_MAX_WRAPPER "Build the Max/MSP wrapper" OFF)
option(BUILD_SC_WRAPPER "Build the SuperCollider wrapper" OFF)
option(BUILD_PYTHON_WRAPPER "Build the Python wrapper" ON)

# --- Testing (Optional, but recommended) ---
option(BUILD_TESTS "Build tests for the project" OFF)

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# macOS specific settings
# Minimum OS X deployment version changed from 10.9 to 10.13 (std::optional::value() is just available for 10.13 and later).
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version")
# Build Universal Binary for Max compatibility (Max externals typically need this)
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Building for Apple Silicon (ARM64)")
        # Adicione aqui suas configurações específicas para ARM64
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Building for Intel (x86_64)")
        # Adicione aqui suas configurações específicas para Intel
    else()
        message(WARNING "Arquitetura do macOS não reconhecida: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()



# --- Output Directories (Consistent for the whole project) ---
set(PROJECT_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
set(PROJECT_LIB_DIR ${PROJECT_DIST_DIR}/lib)
set(PROJECT_BIN_DIR ${PROJECT_DIST_DIR}/bin)
set(PROJECT_TESTS_DIR ${PROJECT_DIST_DIR}/tests)

# Create the directories
file(MAKE_DIRECTORY ${PROJECT_DIST_DIR})
file(MAKE_DIRECTORY ${PROJECT_LIB_DIR})
file(MAKE_DIRECTORY ${PROJECT_BIN_DIR})
file(MAKE_DIRECTORY ${PROJECT_TESTS_DIR})

set(PROJECT_BINARY_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_LIB_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_LIB_DIR})

# --- LibTorch Setup (Centralized) ---
include(FetchContent)

# --- Check for LIBTORCH_PATH environment variable first ---
set(LIBTORCH_PATH_ENV $ENV{LIBTORCH_PATH})

if(LIBTORCH_PATH_ENV AND IS_DIRECTORY "${LIBTORCH_PATH_ENV}")
    message(STATUS "Found LIBTORCH_PATH environment variable: ${LIBTORCH_PATH_ENV}")

    # Check if it contains a valid LibTorch installation
    if(EXISTS "${LIBTORCH_PATH_ENV}/share/cmake/Torch/TorchConfig.cmake")
        message(STATUS "Valid LibTorch installation found at: ${LIBTORCH_PATH_ENV}")
        list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH_ENV}")
        set(Torch_DIR "${LIBTORCH_PATH_ENV}/share/cmake/Torch" CACHE PATH "Path to local Torch CMake config" FORCE)
    else()
        message(WARNING "LIBTORCH_PATH is set but TorchConfig.cmake not found at: ${LIBTORCH_PATH_ENV}/share/cmake/Torch/")
        message(STATUS "Will attempt to find LibTorch elsewhere or download it")
    endif()
endif()

# --- Try to find LibTorch locally first ---
# This allows developers to use a local installation by setting -DTorch_DIR or CMAKE_PREFIX_PATH
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    message(STATUS "LibTorch not found locally. Fetching from remote...")

    # --- Define LibTorch URLs for different platforms ---
    set(LIBTORCH_VERSION "2.8.0")

    if(WIN32)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
    elseif(APPLE)
        # Check for ARM64 architecture
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
        else()
            # URL for x86_64 macOS
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip")
        endif()
    elseif(UNIX)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
    else()
        message(FATAL_ERROR "Unsupported platform for LibTorch download.")
    endif()

    message(STATUS "Downloading LibTorch from: ${LIBTORCH_URL}")

    FetchContent_Declare(
        libtorch
        URL ${LIBTORCH_URL}
    )

    FetchContent_MakeAvailable(libtorch)

    # After fetching, the 'libtorch' directory is available in the build directory
    # The structure is typically ${CMAKE_BINARY_DIR}/_deps/libtorch-src/libtorch
    # We need to tell CMake where to find the TorchConfig.cmake file.
    list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
    set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch" CACHE PATH "Path to fetched Torch CMake config" FORCE)

    # Set Torch_INSTALL_PREFIX for downloaded LibTorch
    set(Torch_INSTALL_PREFIX "${libtorch_SOURCE_DIR}" CACHE PATH "Installation prefix for downloaded Torch libraries" FORCE)

    # Now find_package should work
    find_package(Torch REQUIRED)

else()
    message(STATUS "Found local LibTorch installation: ${Torch_DIR}")

    # Set Torch_INSTALL_PREFIX for local installation if not already set
    if(NOT Torch_INSTALL_PREFIX AND LIBTORCH_PATH_ENV AND IS_DIRECTORY "${LIBTORCH_PATH_ENV}")
        set(Torch_INSTALL_PREFIX "${LIBTORCH_PATH_ENV}" CACHE PATH "Installation prefix for local Torch libraries" FORCE)
        message(STATUS "Set Torch_INSTALL_PREFIX from LIBTORCH_PATH: ${Torch_INSTALL_PREFIX}")
    endif()
endif()

# After find_package, Torch_INSTALL_PREFIX should ideally be set by the Torch CMake files.
# If not, and we have a valid EFFECTIVE_LIBTORCH_PATH, use that.
if(NOT Torch_INSTALL_PREFIX)
    message(STATUS "Torch_INSTALL_PREFIX was not set by find_package(Torch).")

    # Check if we downloaded LibTorch via FetchContent
    if(DEFINED libtorch_SOURCE_DIR AND IS_DIRECTORY "${libtorch_SOURCE_DIR}")
        message(STATUS "Using downloaded LibTorch directory: ${libtorch_SOURCE_DIR}")
        set(Torch_INSTALL_PREFIX ${libtorch_SOURCE_DIR} CACHE PATH "Installation prefix for downloaded Torch libraries" FORCE)
    elseif(EFFECTIVE_LIBTORCH_PATH AND IS_DIRECTORY "${EFFECTIVE_LIBTORCH_PATH}") # Re-check, just in case
        message(STATUS "Attempting to set Torch_INSTALL_PREFIX from EFFECTIVE_LIBTORCH_PATH: ${EFFECTIVE_LIBTORCH_PATH}")
        set(Torch_INSTALL_PREFIX ${EFFECTIVE_LIBTORCH_PATH} CACHE PATH "Installation prefix for Torch libraries (fallback from EFFECTIVE_LIBTORCH_PATH)" FORCE)
    else()
        message(WARNING "Cannot determine Torch_INSTALL_PREFIX from EFFECTIVE_LIBTORCH_PATH either (was '${EFFECTIVE_LIBTORCH_PATH}').")
    endif()
endif()

# Final check for Torch_INSTALL_PREFIX
if(Torch_INSTALL_PREFIX AND IS_DIRECTORY "${Torch_INSTALL_PREFIX}")
    message(STATUS "Torch_INSTALL_PREFIX is confirmed: ${Torch_INSTALL_PREFIX}")

    # Ensure CMAKE_PREFIX_PATH also contains this, as it's used by other find_xxx calls implicitly
    list(PREPEND CMAKE_PREFIX_PATH "${Torch_INSTALL_PREFIX}")
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
    message(STATUS "Updated CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
else()
    message(WARNING "Torch_INSTALL_PREFIX ('${Torch_INSTALL_PREFIX}') is not set or not a valid directory. LibTorch dependency copying will likely fail.")
    message(WARNING "Torch_INSTALL_PREFIX ('${Torch_INSTALL_PREFIX}') is not set or not a valid directory. LibTorch dependency copying will likely fail.")
endif()

# Add Torch include paths
set(TORCH_INCLUDE_DIRS
    "${CMAKE_PREFIX_PATH}/include"
    "${CMAKE_PREFIX_PATH}/include/torch/csrc/api/include"
)

message(STATUS "Torch include dirs:")

foreach(dir ${TORCH_INCLUDE_DIRS})
    message(STATUS " - ${dir}")
endforeach()

# Try to locate key Torch libraries manually if imported target fails
find_library(TORCH_LIB torch PATHS "${CMAKE_PREFIX_PATH}/lib")
find_library(C10_LIB c10 PATHS "${CMAKE_PREFIX_PATH}/lib")
find_library(TORCH_CPU_LIB torch_cpu PATHS "${CMAKE_PREFIX_PATH}/lib")

if(TARGET Torch::Torch)
    message(STATUS "Using imported target Torch::Torch")
else()
    message(STATUS "Torch::Torch target not available. Using manual linking with TORCH_LIBRARIES.")
endif()

message(STATUS "Found LibTorch. Version: ${Torch_VERSION}")

# --- Pre-find Linux SO files and GOMP (do this once) ---
set(LINUX_TORCH_SOS_FILENAMES "") # Initialize empty list of filenames
set(GOMP_LIBRARY_PATH "") # Initialize empty GOMP path

if(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")

    # Use Torch_INSTALL_PREFIX to find the libraries
    if(Torch_INSTALL_PREFIX AND IS_DIRECTORY "${Torch_INSTALL_PREFIX}/lib")
        file(GLOB _linux_torch_sos_filenames RELATIVE "${Torch_INSTALL_PREFIX}/lib" "${Torch_INSTALL_PREFIX}/lib/*.so*")

        if(_linux_torch_sos_filenames)
            set(LINUX_TORCH_SOS_FILENAMES ${_linux_torch_sos_filenames} CACHE INTERNAL "List of Torch .so filenames for Linux") # Store relative paths (filenames)
            message(STATUS "Found Linux Torch SO filenames to copy: ${LINUX_TORCH_SOS_FILENAMES}")
        else()
            message(WARNING "No .so* files found in ${Torch_INSTALL_PREFIX}/lib/ to copy.")
        endif()
    else()
        message(WARNING "Torch_INSTALL_PREFIX/lib directory not found or Torch_INSTALL_PREFIX not set. Cannot list Linux Torch SO files.")
    endif()

    # Find GOMP library once
    find_library(GOMP_LIBRARY_PATH gomp)

    if(GOMP_LIBRARY_PATH)
        set(GOMP_LIBRARY ${GOMP_LIBRARY_PATH}) # Store the found path
        message(STATUS "Linking against OpenMP runtime: ${GOMP_LIBRARY}")
    else()
        set(GOMP_LIBRARY gomp) # Fallback to linking by name
        message(WARNING "libgomp not found by find_library. Linking by name 'gomp'. Torch CPU operations might fail if not found by linker.")
    endif()
endif()

# --- macOS Specific SDK and Libc++ Checks ---
if(APPLE)
    message(STATUS "Configuring for macOS: SDK and libc++ checks")

    # automatically detect Torch dylibs in the Torch_INSTALL_PREFIX/lib directory
    if(Torch_INSTALL_PREFIX AND IS_DIRECTORY "${Torch_INSTALL_PREFIX}/lib")
        file(GLOB _macos_torch_dylibs RELATIVE "${Torch_INSTALL_PREFIX}/lib" "${Torch_INSTALL_PREFIX}/lib/*.dylib")

        if(_macos_torch_dylibs)
            set(MACOS_TORCH_DYLIBS_FILENAMES ${_macos_torch_dylibs} CACHE INTERNAL "List of Torch .dylib filenames for macOS")
            message(STATUS "Auto-detected Torch dylibs for macOS: ${MACOS_TORCH_DYLIBS_FILENAMES}")
        else()
            message(WARNING "No .dylib files found in ${Torch_INSTALL_PREFIX}/lib/ to copy.")
        endif()
    else()
        # Fallback: try to use the libtorch_SOURCE_DIR if available (for downloaded LibTorch)
        if(DEFINED libtorch_SOURCE_DIR AND IS_DIRECTORY "${libtorch_SOURCE_DIR}/lib")
            file(GLOB _macos_torch_dylibs RELATIVE "${libtorch_SOURCE_DIR}/lib" "${libtorch_SOURCE_DIR}/lib/*.dylib")

            if(_macos_torch_dylibs)
                set(MACOS_TORCH_DYLIBS_FILENAMES ${_macos_torch_dylibs} CACHE INTERNAL "List of Torch .dylib filenames for macOS")
                message(STATUS "Auto-detected Torch dylibs for macOS (fallback): ${MACOS_TORCH_DYLIBS_FILENAMES}")
            else()
                message(WARNING "No .dylib files found in ${libtorch_SOURCE_DIR}/lib/ to copy.")
            endif()
        else()
            message(WARNING "Torch_INSTALL_PREFIX/lib directory not found or Torch_INSTALL_PREFIX not set. Cannot list macOS Torch dylib files.")
        endif()
    endif()

    # Automatically detect the active macOS SDK path using xcrun
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE OSX_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(OSX_SDK_PATH AND EXISTS "${OSX_SDK_PATH}")
        set(CMAKE_OSX_SYSROOT ${OSX_SDK_PATH} CACHE PATH "Path to macOS SDK" FORCE)
        message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")
    else()
        message(WARNING "Could not determine macOS SDK path using xcrun.")
    endif()

    # Check whether standard C++ headers (e.g., <cstddef>) are found automatically
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
        #include <cstddef>
        int main() { std::size_t x = 0; return x; }
    " LIBCXX_HEADERS_FOUND)

    if(NOT LIBCXX_HEADERS_FOUND AND CMAKE_OSX_SYSROOT AND EXISTS "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
        message(WARNING "Standard libc++ headers not found automatically. Adding SDK's libc++ include directory.")
        include_directories(SYSTEM "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")

    # Forcing for all CXX targets might be too broad, consider target_include_directories if needed
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
    elseif(LIBCXX_HEADERS_FOUND)
        message(STATUS "Standard libc++ headers found automatically.")
    else()
        message(WARNING "Standard libc++ headers not found, and SDK path for libc++ is not available or invalid.")
    endif()
endif()

# --- Global Helper Function for Copying Dependencies ---
function(add_copy_dependency_command)
    set(options)
    set(oneValueArgs TARGET_NAME SOURCE_FILE DESTINATION_DIR LIB_FILENAME MESSAGE)
    set(multiValueArgs)
    cmake_parse_arguments(COPY_CMD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT COPY_CMD_TARGET_NAME OR NOT COPY_CMD_SOURCE_FILE OR NOT COPY_CMD_DESTINATION_DIR)
        message(FATAL_ERROR "add_copy_dependency_command: TARGET_NAME, SOURCE_FILE, and DESTINATION_DIR are required.")
    endif()

    if(NOT IS_ABSOLUTE "${COPY_CMD_DESTINATION_DIR}")
        message(FATAL_ERROR "add_copy_dependency_command: DESTINATION_DIR must be an absolute path. Got: ${COPY_CMD_DESTINATION_DIR}")
    endif()

    # If LIB_FILENAME is not provided, derive it from SOURCE_FILE
    if(NOT COPY_CMD_LIB_FILENAME)
        get_filename_component(COPY_CMD_LIB_FILENAME ${COPY_CMD_SOURCE_FILE} NAME)
    endif()

    # Default message if not provided
    if(NOT COPY_CMD_MESSAGE)
        set(COPY_CMD_MESSAGE "Copying dependency ${COPY_CMD_LIB_FILENAME} to ${COPY_CMD_DESTINATION_DIR} for ${COPY_CMD_TARGET_NAME}")
    endif()

    add_custom_command(
        TARGET ${COPY_CMD_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${COPY_CMD_SOURCE_FILE}"
        "${COPY_CMD_DESTINATION_DIR}/${COPY_CMD_LIB_FILENAME}"
        COMMENT "${COPY_CMD_MESSAGE}"
        VERBATIM
    )
endfunction()

# --- Include Directories (Public headers for the project) ---
include_directories(include) # headers for the project

# --- Submodules / Third-Party Dependencies ---
# Note: If these are actual Git submodules, ensure they are initialized and updated.
# If they provide CMake targets (e.g., via add_subdirectory in their own CMakeLists.txt
# or through find_package), you might not need to do much more here.
# Otherwise, you might need to use add_subdirectory pointing to their source and build them directly.

# Example for pd.cmake (if it's a CMake utility module, not a library to build)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pd.cmake)

# Example for max-sdk-base (if it needs to be built or its headers included)
# add_subdirectory(third_party/max-sdk-base)
# include_directories(third_party/max-sdk-base/c74support/max-includes) # Or however it exposes includes

# --- Project Source Code Subdirectories ---
add_subdirectory(core) # Changed from src/contorchionist_core
add_subdirectory(third_party/pybind11)
add_subdirectory(wrappers) # Changed from src/wrappers

if(BUILD_TESTS)
    enable_testing() # Enables CTest
    add_subdirectory(tests)
endif()

# --- Default Build Type ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

# --- Copy LibTorch dylibs to output directories ---
if(APPLE)
    message(STATUS "Setting up LibTorch dylib copying for macOS")

    # Determine LibTorch lib directory
    set(LIBTORCH_LIB_DIR "")

    # Priority 1: Downloaded LibTorch
    if(DEFINED libtorch_SOURCE_DIR AND IS_DIRECTORY "${libtorch_SOURCE_DIR}/lib")
        set(LIBTORCH_LIB_DIR "${libtorch_SOURCE_DIR}/lib")
        message(STATUS "Using downloaded LibTorch lib directory: ${LIBTORCH_LIB_DIR}")

    # Priority 2: Local installation from LIBTORCH_PATH
    elseif(LIBTORCH_PATH_ENV AND IS_DIRECTORY "${LIBTORCH_PATH_ENV}/lib")
        set(LIBTORCH_LIB_DIR "${LIBTORCH_PATH_ENV}/lib")
        message(STATUS "Using local LibTorch lib directory: ${LIBTORCH_LIB_DIR}")

    # Priority 3: Torch_INSTALL_PREFIX
    elseif(Torch_INSTALL_PREFIX AND IS_DIRECTORY "${Torch_INSTALL_PREFIX}/lib")
        set(LIBTORCH_LIB_DIR "${Torch_INSTALL_PREFIX}/lib")
        message(STATUS "Using Torch_INSTALL_PREFIX lib directory: ${LIBTORCH_LIB_DIR}")
    endif()

    # Only proceed if we found a valid LibTorch lib directory
    if(LIBTORCH_LIB_DIR AND IS_DIRECTORY "${LIBTORCH_LIB_DIR}")
        # This target is no longer needed as copying is handled per-wrapper
        # add_custom_target(copy_libtorch_dylibs ALL
        #     COMMENT "Copying LibTorch dylibs to output directories"
        # )
    else()
        message(WARNING "Could not determine LibTorch lib directory for dylib copying")
    endif()
endif()

# --- Global Resource Copying ---
set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/resources)
if(EXISTS ${RESOURCES_DIR})
    set(WRAPPER_DIST_DIRS
        ${PROJECT_DIST_DIR}/puredata
        ${PROJECT_DIST_DIR}/max
        ${PROJECT_DIST_DIR}/python
    )

    # Create a single target that handles all resource copying
    add_custom_target(copy_global_resources ALL COMMENT "Copying global resources to wrapper directories")

    foreach(DIST_DIR ${WRAPPER_DIST_DIRS})
        # The copy command for each wrapper will be triggered by the parent target
        add_custom_command(
            TARGET copy_global_resources
            COMMAND ${CMAKE_COMMAND} -E make_directory ${DIST_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR} ${DIST_DIR}/resources
            COMMENT "Copying resources to ${DIST_DIR}"
            VERBATIM
        )
    endforeach()
    message(STATUS "Configured copying of 'resources' directory to all wrapper distribution directories.")
endif()

message(STATUS "Contorchionist project configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build directory: ${PROJECT_BINARY_DIR}")