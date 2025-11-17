# CMakeLists.txt for contorchionist_core
cmake_minimum_required(VERSION 3.15)

project(contorchionist_core LANGUAGES CXX C)

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # PIC is usually default for SHARED libs but good to have

# For shared libraries, especially on GCC/Clang, control symbol visibility
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# --- Output Directories ---
# Use project-wide directories defined in the root CMakeLists.txt
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_LIB_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_LIB_DIR}")

# --- macOS Specific Settings ---
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version")
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "macOS Architectures (e.g., arm64;x86_64 or x86_64)")
    message(STATUS "Configuring for macOS. Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}, Architectures: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# --- LibTorch Setup ---
# LibTorch is now found by the root CMakeLists.txt.
# This target simply uses the already-found package.
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Torch package not found. It should have been found by the root CMakeLists.txt")
endif()

# --- Source Files ---
# Automatically include all .cpp files in src/ that don't start with _
file(GLOB_RECURSE CONTORCHIONIST_CORE_SOURCES "src/*.cpp")
list(FILTER CONTORCHIONIST_CORE_SOURCES EXCLUDE REGEX "src/_.*\\.cpp$")

# --- Header Files ---
# Automatically include all .h files in include/ that don't start with _
file(GLOB_RECURSE CONTORCHIONIST_CORE_HEADERS "include/*.h")
list(FILTER CONTORCHIONIST_CORE_HEADERS EXCLUDE REGEX "include/_.*\\.h$")

# --- Library Definition ---
add_library(contorchionist_core SHARED ${CONTORCHIONIST_CORE_SOURCES}) # Changed STATIC to SHARED
add_library(contorchionist::core ALIAS contorchionist_core)

# --- Symbol Export ---
include(GenerateExportHeader)

# Creates a header (e.g., contorchionist_core_export.h) with macros for symbol visibility
# Base name is the target name, defines CONTORCHIONIST_CORE_EXPORT, CONTORCHIONIST_CORE_NO_EXPORT etc.
generate_export_header(contorchionist_core
    BASE_NAME CONTORCHIONIST_CORE # Results in CONTORCHIONIST_CORE_EXPORT macro
    EXPORT_MACRO_NAME CONTORCHIONIST_CORE_EXPORT
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/contorchionist_core/contorchionist_core_export.h"

    # DEPRECATED_MACRO_NAME CONTORCHIONIST_CORE_DEPRECATED # Optional
)

# Add the directory containing the generated export header to include paths for contorchionist_core itself
target_include_directories(contorchionist_core PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/include"
)

# And for consumers (PUBLIC) so they can see the export macros if needed by headers
target_include_directories(contorchionist_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # For export header during build
    $<INSTALL_INTERFACE:include> # For export header when installed (will be installed later)
)

# --- RPATH Settings ---
if(APPLE)
    set(_rpath "@loader_path")
elseif(UNIX)
    set(_rpath "$ORIGIN")
else()
    set(_rpath "")
endif()

# --- Target Properties ---
set(CONTORCHIONIST_CORE_VERSION "1.0.0") # Define your library's version (used for package config)

# Convert absolute paths to relative paths for PUBLIC_HEADER
set(PUBLIC_HEADER_LIST "")

foreach(HEADER ${CONTORCHIONIST_CORE_HEADERS})
    file(RELATIVE_PATH REL_HEADER ${CMAKE_CURRENT_SOURCE_DIR} ${HEADER})
    list(APPEND PUBLIC_HEADER_LIST ${REL_HEADER})
endforeach()

set_target_properties(contorchionist_core PROPERTIES
    PUBLIC_HEADER "${PUBLIC_HEADER_LIST}"
    INSTALL_RPATH "${_rpath}" # Add RPATH for the shared library
    MACOSX_RPATH TRUE # Ensure RPATH is processed for the library itself
    BUILD_WITH_INSTALL_RPATH TRUE # Uses INSTALL_RPATH also in the build tree
    INSTALL_NAME_DIR "@rpath" # Sets the directory part of the install name for the library
)

# --- Include Directories ---
target_include_directories(contorchionist_core
    PUBLIC
    $<INSTALL_INTERFACE:include> # For installed headers (e.g. <contorchionist_core/rms_analyzer.h>)
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # For headers in source tree during build (e.g. <header.h>)
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # For export header during build
    ${Torch_INCLUDE_DIRS} # Publicly expose Torch include directories
    PRIVATE

    # Headers are in include/, sources are in src/
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- Linking ---
if(TARGET Torch::Torch)
    message(STATUS "Linking contorchionist_core with imported target Torch::Torch")
    target_link_libraries(contorchionist_core PUBLIC Torch::Torch)
elseif(TORCH_LIBRARIES)
    message(STATUS "Linking contorchionist_core with TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    target_link_libraries(contorchionist_core PUBLIC ${TORCH_LIBRARIES})
else()
    message(FATAL_ERROR "Could not determine Torch libraries to link against. Neither Torch::Torch target nor TORCH_LIBRARIES variable is available.")
endif()

# --- Default Build Type ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type (e.g., Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    message(STATUS "Setting default CMAKE_BUILD_TYPE to Release.")
endif()

# --- Installation ---
include(GNUInstallDirs)
install(TARGETS contorchionist_core
    EXPORT contorchionist_coreTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT contorchionist_core_Runtime # For DLLs on Windows
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT contorchionist_core_Libraries # For .so, .dylib

    # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT contorchionist_core_Libraries # Less relevant for SHARED, but can be kept for import libs on Windows
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/contorchionist_core COMPONENT contorchionist_core_Headers
)

# Install the generated export header
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/contorchionist_core/contorchionist_core_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/contorchionist_core COMPONENT contorchionist_core_Headers
)

install(EXPORT contorchionist_coreTargets
    FILE contorchionist_coreTargets.cmake
    NAMESPACE contorchionist::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/contorchionist_core
    COMPONENT contorchionist_core_Development
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/contorchionist_coreConfigVersion.cmake"
    VERSION ${CONTORCHIONIST_CORE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Create the contorchionist_coreConfig.cmake file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/contorchionist_coreConfig.cmake"
    "include(CMakeFindDependencyMacro)
# contorchionist_core's public interface depends on Torch (e.g. Torch headers are in public contorchionist_core headers)
find_dependency(Torch REQUIRED)

if(NOT TARGET contorchionist::core)
    include(\"${CMAKE_CURRENT_LIST_DIR}/contorchionist_coreTargets.cmake\")
endif()
"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/contorchionist_coreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/contorchionist_coreConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/contorchionist_core
    COMPONENT contorchionist_core_Development
)

message(STATUS "contorchionist_core configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "contorchionist_core shared library will be built in: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} (runtime in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})")
message(STATUS "Consumers of contorchionist_core are responsible for ensuring LibTorch shared libraries are available at runtime.")
message(STATUS "To install contorchionist_core, after building, run: cmake --install . --prefix <your_install_path> or your build system's install command.")