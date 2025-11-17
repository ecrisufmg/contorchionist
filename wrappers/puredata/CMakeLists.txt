cmake_minimum_required(VERSION 3.18)
project(contorchionist_pd CXX)

# --- pd.cmake Integration ---
include(FetchContent)

# --- Try to find Pure Data locally first ---
if(NOT DEFINED PD_SOURCES_PATH)
    find_path(PD_HEADER_PATH m_pd.h
        HINTS
        "/usr/include/pd"
        "/usr/local/include/pd"
        "$ENV{HOME}/pd-src/src"
        "C:/Program Files/Pd/src"
        "C:/Program Files/Pd64/src"
        "/Applications/Pd.app/Contents/Resources/src"
    )

    if(PD_HEADER_PATH)
        get_filename_component(PD_SOURCES_PATH ${PD_HEADER_PATH} DIRECTORY)
        message(STATUS "Found local Pure Data source: ${PD_SOURCES_PATH}")
    else()
        message(STATUS "Pure Data source not found locally. Fetching from remote...")
        FetchContent_Declare(
            pd_source
            GIT_REPOSITORY https://github.com/pure-data/pure-data.git
            GIT_TAG 0.55-2
        )
        FetchContent_MakeAvailable(pd_source)
        set(PD_SOURCES_PATH "${pd_source_SOURCE_DIR}/src" CACHE PATH "Path to fetched Pure Data source" FORCE)
    endif()
endif()

# Set the path to the pd.cmake directory (relative to the project root)
set(PDCMAKE_DIR ${CMAKE_SOURCE_DIR}/third_party/pd.cmake CACHE PATH "Path to pd.cmake directory")
include(${PDCMAKE_DIR}/pd.cmake)

message(STATUS "Pure Data Wrapper CMakeLists.txt (contorchionist project):")
message(STATUS "  PD_LIBRARY: ${PD_LIBRARY}")
message(STATUS "  PD_INCLUDE_DIR: ${PD_INCLUDE_DIR}")
message(STATUS "  PD_EXTERNAL_SUFFIX: ${PD_EXTERNAL_SUFFIX}")

# --- Common Setup & Utility Definitions ---

# Define a single output directory for all externals
set(PD_EXTERNAL_OUTPUT_DIR "${PROJECT_DIST_DIR}/puredata")
file(MAKE_DIRECTORY ${PD_EXTERNAL_OUTPUT_DIR})

# Define RPATH logic once
if(APPLE)
    set(_rpath "@loader_path")
elseif(UNIX)
    set(_rpath "$ORIGIN")
else()
    set(_rpath "")
endif()

# Check for utility sources once
set(PD_ARG_PARSER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/pd_arg_parser.cpp")
set(PD_ARG_PARSER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils/include")
set(USE_PD_ARG_PARSER FALSE)

if(EXISTS "${PD_ARG_PARSER_SRC}" AND IS_DIRECTORY "${PD_ARG_PARSER_INCLUDE_DIR}")
    set(USE_PD_ARG_PARSER TRUE)
    message(STATUS "Found pd_arg_parser utility, will be linked where needed.")
endif()

set(PD_TORCH_DEVICE_ADAPTER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/pd_torch_device_adapter.cpp")
set(PD_TORCH_DEVICE_ADAPTER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils/include")
set(USE_PD_TORCH_DEVICE_ADAPTER FALSE)

if(EXISTS "${PD_TORCH_DEVICE_ADAPTER_SRC}" AND IS_DIRECTORY "${PD_TORCH_DEVICE_ADAPTER_INCLUDE_DIR}")
    set(USE_PD_TORCH_DEVICE_ADAPTER TRUE)
    message(STATUS "Found pd_torch_device_adapter utility, will be linked where needed.")
endif()

set(PD_TORCH_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/pd_torch_utils.cpp")
set(PD_TORCH_UTILS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils/include")
set(USE_PD_TORCH_UTILS FALSE)

if(EXISTS "${PD_TORCH_UTILS_SRC}" AND IS_DIRECTORY "${PD_TORCH_UTILS_INCLUDE_DIR}")
    set(USE_PD_TORCH_UTILS TRUE)
    message(STATUS "Found pd_torch_utils utility, will be linked where needed.")
endif()

# --- Reusable Function to Create a Pure Data External ---
#
# This function automates the creation of a PD external from a source file.
# It handles name conversion, target creation, linking, and property setting.
#
function(add_puredata_external SOURCE_FILE PD_EXT_NAME)
    # 1. The external name is now passed directly into the function.

    # 2. Create a valid CMake target name from the correct external name
    string(REPLACE "~" "_tilde" CMAKE_TARGET_NAME_TEMP "${PD_EXT_NAME}")
    string(REPLACE "." "_" CMAKE_TARGET_NAME_BASE "${CMAKE_TARGET_NAME_TEMP}")
    set(CMAKE_TARGET_NAME "pd_${CMAKE_TARGET_NAME_BASE}")

    message(STATUS "--- Configuring PD External: ${PD_EXT_NAME} ---")
    message(STATUS "  Source: ${SOURCE_FILE}")
    message(STATUS "  CMake Target: ${CMAKE_TARGET_NAME}") # This will now be unique and correct

    # 3. Create the external target using the pd.cmake function
    pd_add_external(${PD_EXT_NAME} ${SOURCE_FILE} TARGET ${CMAKE_TARGET_NAME})

    # 4. Determine if this external needs pd_torch_utils by reading the source file
    set(NEEDS_PD_TORCH_UTILS FALSE)

    if(EXISTS "${SOURCE_FILE}")
        file(READ "${SOURCE_FILE}" SOURCE_CONTENT)

        if(SOURCE_CONTENT MATCHES "PdTorchUtils::" OR SOURCE_CONTENT MATCHES "#include.*pd_torch_utils")
            set(NEEDS_PD_TORCH_UTILS TRUE)
        endif()
    endif()

    # 5. Add common utility sources if they exist
    if(USE_PD_ARG_PARSER)
        target_sources(${CMAKE_TARGET_NAME} PRIVATE ${PD_ARG_PARSER_SRC})
    endif()

    if(USE_PD_TORCH_DEVICE_ADAPTER)
        target_sources(${CMAKE_TARGET_NAME} PRIVATE ${PD_TORCH_DEVICE_ADAPTER_SRC})
    endif()

    if(USE_PD_TORCH_UTILS AND NEEDS_PD_TORCH_UTILS)
        target_sources(${CMAKE_TARGET_NAME} PRIVATE ${PD_TORCH_UTILS_SRC})
        message(STATUS "  Added pd_torch_utils.cpp to ${PD_EXT_NAME} (detected usage)")
    endif()

    # 6. Add common include directories
    target_include_directories(${CMAKE_TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )

    if(USE_PD_ARG_PARSER)
        target_include_directories(${CMAKE_TARGET_NAME} PRIVATE ${PD_ARG_PARSER_INCLUDE_DIR})
    endif()

    if(USE_PD_TORCH_DEVICE_ADAPTER)
        target_include_directories(${CMAKE_TARGET_NAME} PRIVATE ${PD_TORCH_DEVICE_ADAPTER_INCLUDE_DIR})
    endif()

    if(USE_PD_TORCH_UTILS AND NEEDS_PD_TORCH_UTILS)
        target_include_directories(${CMAKE_TARGET_NAME} PRIVATE ${PD_TORCH_UTILS_INCLUDE_DIR})
    endif()

    # 7. Link against the core library
    if(TARGET contorchionist::core)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE contorchionist::core)
    elseif(TARGET contorchionist_core)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE contorchionist_core)
    else()
        message(WARNING "Target 'contorchionist_core' not found. Cannot link ${CMAKE_TARGET_NAME}.")
    endif()

    # 8. Set output properties
    set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${PD_EXT_NAME}
        INSTALL_RPATH "${_rpath}"
        BUILD_WITH_INSTALL_RPATH TRUE # ENURE RPATH IS SET
        LIBRARY_OUTPUT_DIRECTORY "${PD_EXTERNAL_OUTPUT_DIR}"
    )

    # 9. Add dependencies to common library copy targets
    # if(TARGET pd_copy_torch_libs)
    #     add_dependencies(${CMAKE_TARGET_NAME} pd_copy_torch_libs)
    # endif()

    # if(TARGET pd_copy_contorchionist_core_lib)
    #     add_dependencies(${CMAKE_TARGET_NAME} pd_copy_contorchionist_core_lib)
    # endif()

    copy_core_libs_to_wrapper_dist(
        TARGET_NAME ${CMAKE_TARGET_NAME}
        DIST_DIR ${PD_EXTERNAL_OUTPUT_DIR}
    )

    # 10. Link against pd_global_state for externals that need shared state
    set(EXTERNALS_NEEDING_GLOBAL_STATE "torch.activation" "torch.linear" "torch.mha" "torch.reshape" "torch.sequential" "torch.ls2tensor")

    if(PD_EXT_NAME IN_LIST EXTERNALS_NEEDING_GLOBAL_STATE)
        if(TARGET pd_global_state)
            target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE pd_global_state)
            add_dependencies(${CMAKE_TARGET_NAME} pd_global_state)

            # Ensure pd_global_state is copied before this external is built
            # if(TARGET pd_copy_contorchionist_core_lib)
            #     add_dependencies(pd_global_state pd_copy_contorchionist_core_lib)
            # endif()

            message(STATUS "${PD_EXT_NAME} linked with pd_global_state (shared state coordination)")
        else()
            message(STATUS "pd_global_state not available for ${PD_EXT_NAME}")
        endif()
    endif()

    message(STATUS "--- Finished configuring ${PD_EXT_NAME} ---")
endfunction()

# --- Configure the pd_global_state library ---
message(STATUS "--- Configuring pd_global_state Library ---")

# Check if the required source and header files exist
set(PD_GLOBAL_STATE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/pd_global_state.cpp")
set(PD_GLOBAL_STATE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/utils/include/pd_global_state.h")

if(EXISTS "${PD_GLOBAL_STATE_SRC}")
    message(STATUS "✓ Found pd_global_state source: ${PD_GLOBAL_STATE_SRC}")

    # Create the shared library
    add_library(pd_global_state SHARED "${PD_GLOBAL_STATE_SRC}")

    # Include directories (same approach as pd.cmake uses)
    target_include_directories(pd_global_state PRIVATE
        ${PD_SOURCES_PATH} # For Pure Data headers (m_pd.h)
    )

    target_include_directories(pd_global_state PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/include" # For pd_global_state.h
        "${CMAKE_SOURCE_DIR}/core/include" # For core library headers
    )

    # Add compile definitions (same as pd.cmake adds to externals)
    if(PD_FLOATSIZE STREQUAL 64)
        target_compile_definitions(pd_global_state PRIVATE PD_FLOATSIZE=64)
    endif()

    # Configure for Pure Data dynamic loading
    # This allows undefined symbols (like pd_error, post) to be resolved at runtime
    if(APPLE)
        target_link_options(pd_global_state PRIVATE "-undefined" "dynamic_lookup")
    elseif(UNIX AND NOT APPLE)
        target_link_options(pd_global_state PRIVATE "-Wl,--allow-shlib-undefined")
    endif()

    # Link against the core library
    if(TARGET contorchionist_core)
        target_link_libraries(pd_global_state PRIVATE contorchionist_core)
        message(STATUS "✓ pd_global_state linked with contorchionist_core")
    else()
        message(WARNING "contorchionist_core target not found. pd_global_state may not function correctly.")
    endif()

    # Set library properties
    set_target_properties(pd_global_state PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PD_EXTERNAL_OUTPUT_DIR}"
        INSTALL_RPATH "${_rpath}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    copy_core_libs_to_wrapper_dist(
        TARGET_NAME pd_global_state
        DIST_DIR ${PD_EXTERNAL_OUTPUT_DIR}
    )

    message(STATUS "✓ pd_global_state library configured successfully")
    message(STATUS "  Output directory: ${PD_EXTERNAL_OUTPUT_DIR}")
    message(STATUS "  RPATH: ${_rpath}")

else()
    message(WARNING "pd_global_state source file not found at: ${PD_GLOBAL_STATE_SRC}")
    message(WARNING "pd_global_state library will NOT be created.")
endif()

message(STATUS "--- Finished Configuring pd_global_state ---")

# --- Automatic Discovery and Creation of PD Externals ---
# Find all potential source files in the src directory
file(GLOB PD_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Loop through each found source file and create a target for it
foreach(SOURCE_FILE ${PD_SOURCE_FILES})
    get_filename_component(FILENAME ${SOURCE_FILE} NAME)

    # RULE: Ignore files starting with an underscore "_"
    if(NOT FILENAME MATCHES "^_.*")
        # Robustly extract the external name by removing the .cpp extension
        string(REGEX REPLACE "\\.cpp$" "" PD_EXT_NAME_FOR_TARGET ${FILENAME})

        # Call our helper function with the source file AND the correct external name
        add_puredata_external(${SOURCE_FILE} ${PD_EXT_NAME_FOR_TARGET})
    else()
        message(STATUS "Skipping test/dev file: ${FILENAME}")
    endif()
endforeach()

# --- Common Dependency Copy Targets (REMOVED) ---
# This logic is now handled by the copy_core_libs_to_wrapper_dist function.

# --- Global Help Directory Copy ---
set(HELP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/help")

if(IS_DIRECTORY "${HELP_SOURCE_DIR}" AND NOT TARGET pd_copy_all_help_files)
    add_custom_target(pd_copy_all_help_files ALL
        COMMENT "Preparing to copy entire help directory to ${PD_EXTERNAL_OUTPUT_DIR}"
    )

    # Copy the entire help directory structure to the output directory
    add_custom_command(
        TARGET pd_copy_all_help_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${HELP_SOURCE_DIR}" "${PD_EXTERNAL_OUTPUT_DIR}"
        COMMENT "Copying entire help directory structure from ${HELP_SOURCE_DIR} to ${PD_EXTERNAL_OUTPUT_DIR}"
        VERBATIM
    )

    message(STATUS "Configured global copy of entire help directory structure from ${HELP_SOURCE_DIR} to ${PD_EXTERNAL_OUTPUT_DIR}")
else()
    if(NOT IS_DIRECTORY "${HELP_SOURCE_DIR}")
        message(WARNING "Help directory not found: ${HELP_SOURCE_DIR}")
    endif()
endif()

message(STATUS "======== Finished Configuring PureData Wrappers ========")