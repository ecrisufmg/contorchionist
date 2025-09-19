cmake_minimum_required(VERSION 3.18)
project(contorchionist_max CXX)

if(NOT(WIN32 OR APPLE))
    message(WARNING "! As you are building in ${CMAKE_SYSTEM_NAME}, the Max external will not be built. Only Windows and macOS are supported for Max externals at this time.")
    return()
endif()

# --- Max SDK Setup ---
set(MAX_SDK_PATH ${CMAKE_SOURCE_DIR}/third_party/max-sdk-base CACHE PATH "Path to Max SDK base directory")
message(STATUS "Using Max SDK from: ${MAX_SDK_PATH}")

# Define MAX_SDK_BASE_SCRIPT_PATH
set(MAX_SDK_BASE_SCRIPT_PATH "${MAX_SDK_PATH}/script")
message(STATUS "Max SDK script path: ${MAX_SDK_BASE_SCRIPT_PATH}")

# --- Architecture Setup for macOS ---
if(APPLE)
    # Check LibTorch architecture and set compatible architecture
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE NATIVE_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # For Apple Silicon Macs, use arm64; for Intel Macs, use x86_64
    if(NATIVE_ARCH STREQUAL "arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build for Apple Silicon" FORCE)
        message(STATUS "Set CMake architecture for Apple Silicon: ${CMAKE_OSX_ARCHITECTURES}")
    else()
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build for Intel" FORCE)
        message(STATUS "Set CMake architecture for Intel: ${CMAKE_OSX_ARCHITECTURES}")
    endif()

    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)

    # Note: Skipping Max linker flags as they're causing issues with our build
    # The Max frameworks should provide the necessary symbol resolution
    message(STATUS "Skipping Max linker flags - using framework-based linking only")
endif()

# Include the Max SDK pre-target script
include(${MAX_SDK_BASE_SCRIPT_PATH}/max-pretarget.cmake)

# --- Common Setup ---
set(MAX_EXTERNAL_OUTPUT_DIR "${PROJECT_DIST_DIR}/max")
file(MAKE_DIRECTORY ${MAX_EXTERNAL_OUTPUT_DIR})

# Determine Max external bundle extension
if(WIN32)
    set(MAX_EXTERN_BUNDLE_EXTENSION "mxe64")
elseif(APPLE)
    set(MAX_EXTERN_BUNDLE_EXTENSION "mxo")
else()
    set(MAX_EXTERN_BUNDLE_EXTENSION "mxe")
    message(WARNING "Unsupported OS for Max external bundle extension, using .mxe as default.")
endif()

# Find Max SDK libraries and frameworks
if(WIN32)
    find_library(MAX_API_LIBRARY_PATH
        NAMES MaxAPI MaxAPI.lib
        HINTS "${MAX_SDK_PATH}/c74support/max-includes/x64"
        REQUIRED)

    find_library(MAX_AUDIO_LIBRARY_PATH
        NAMES MaxAudio MaxAudio.lib
        HINTS "${MAX_SDK_PATH}/c74support/msp-includes/x64"
    )

    if(MAX_API_LIBRARY_PATH)
        message(STATUS "Found MaxAPI Library: ${MAX_API_LIBRARY_PATH}")
    endif()

    if(MAX_AUDIO_LIBRARY_PATH)
        message(STATUS "Found MaxAudio Library: ${MAX_AUDIO_LIBRARY_PATH}")
    endif()
elseif(APPLE)
    # Find Max frameworks for macOS
    find_library(MAX_API_FRAMEWORK MaxAPI
        HINTS "${MAX_SDK_PATH}/c74support/max-includes"
    )
    find_library(MAX_AUDIO_FRAMEWORK MaxAudioAPI
        HINTS "${MAX_SDK_PATH}/c74support/msp-includes"
    )

    if(MAX_API_FRAMEWORK)
        message(STATUS "Found MaxAPI Framework: ${MAX_API_FRAMEWORK}")
    endif()

    if(MAX_AUDIO_FRAMEWORK)
        message(STATUS "Found MaxAudioAPI Framework: ${MAX_AUDIO_FRAMEWORK}")
    endif()
endif()

# --- Reusable Function to Create a Max External ---
function(add_max_external SOURCE_FILE)
    get_filename_component(FULL_BASENAME ${SOURCE_FILE} NAME)
    string(REGEX REPLACE "\\.cpp$" "" RAW_BASENAME "${FULL_BASENAME}")
    string(REPLACE "~" "_tilde" TARGET_NAME_SUFFIX ${RAW_BASENAME})
    set(CMAKE_TARGET_NAME "max_${TARGET_NAME_SUFFIX}")

    message(STATUS "--- Configuring Max External: ${RAW_BASENAME} ---")
    message(STATUS "  Source: ${SOURCE_FILE}")
    message(STATUS "  CMake Target: ${CMAKE_TARGET_NAME}")

    add_library(${CMAKE_TARGET_NAME} MODULE ${SOURCE_FILE})

    # --- Include Directories ---
    target_include_directories(${CMAKE_TARGET_NAME} PRIVATE
        "${MAX_SDK_PATH}/c74support/max-includes"
        "${MAX_SDK_PATH}/c74support/msp-includes"
        "${CMAKE_SOURCE_DIR}/core/include"
    )

    # --- Linking ---
    if(TARGET contorchionist_core)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE contorchionist_core)
    else()
        message(WARNING "contorchionist_core target not found for ${CMAKE_TARGET_NAME}")
    endif()

    if(Torch_FOUND)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})
    endif()

    # --- Max SDK Libraries ---
    if(WIN32)
        if(MAX_API_LIBRARY_PATH)
            target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE "${MAX_API_LIBRARY_PATH}")
        endif()

        if(MAX_AUDIO_LIBRARY_PATH)
            target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE "${MAX_AUDIO_LIBRARY_PATH}")
        endif()
    elseif(APPLE)
        # For macOS, MSP externals need MaxAudioAPI.framework for audio symbols
        if(MAX_AUDIO_FRAMEWORK)
            target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE ${MAX_AUDIO_FRAMEWORK})
        endif()

        # Link against required system frameworks
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE
            "-framework CoreFoundation"
            "-framework CoreServices"
            "-framework CoreAudio"
            "-framework AudioUnit"
        )

        # Set proper rpath for Max frameworks and libraries
        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path/../../../../Frameworks;@loader_path/../../.."
        )
    endif()

    # --- Apply Max SDK naming and properties ---
    set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${RAW_BASENAME}"
        SUFFIX ".${MAX_EXTERN_BUNDLE_EXTENSION}"
        RUNTIME_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
        BUNDLE_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
    )

    if(APPLE)
        set(DEFAULT_INFO_PLIST_PATH "${MAX_SDK_PATH}/script/Info.plist.in")

        if(NOT EXISTS ${DEFAULT_INFO_PLIST_PATH})
            message(FATAL_ERROR "Info.plist default for macOS not found at ${DEFAULT_INFO_PLIST_PATH}.")
        endif()

        # Process the Info.plist template
        set(AUTHOR_DOMAIN "com.cycling74")
        set(BUNDLE_IDENTIFIER "max.${RAW_BASENAME}")
        set(PACKAGE_VERSION "1.0")
        set(COPYRIGHT_STRING "Max External")
        set(EXCLUDE_FROM_COLLECTIVES "")

        configure_file(
            "${DEFAULT_INFO_PLIST_PATH}"
            "${CMAKE_CURRENT_BINARY_DIR}/Info.plist.${RAW_BASENAME}"
            @ONLY
        )

        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
            FRAMEWORK TRUE
            BUNDLE TRUE
            BUNDLE_EXTENSION "${MAX_EXTERN_BUNDLE_EXTENSION}"
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist.${RAW_BASENAME}"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.cycling74.max.${RAW_BASENAME}"
            MACOSX_BUNDLE_BUNDLE_NAME "${RAW_BASENAME}"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
            MACOSX_BUNDLE_BUNDLE_VERSION "1.0"

            # Force output to our desired directory
            RUNTIME_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${MAX_EXTERNAL_OUTPUT_DIR}"
        )

        # Add post-build code signing for Apple Silicon compatibility
        add_custom_command(TARGET ${CMAKE_TARGET_NAME} POST_BUILD
            COMMAND codesign --force --deep -s - "$<TARGET_BUNDLE_DIR:${CMAKE_TARGET_NAME}>"
            COMMENT "Code signing ${RAW_BASENAME} for macOS security policy compliance"
            VERBATIM
        )

        # Add a post-build step to ensure the bundle ends up in the right place
        add_custom_command(TARGET ${CMAKE_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Moving ${RAW_BASENAME}.${MAX_EXTERN_BUNDLE_EXTENSION} to ${MAX_EXTERNAL_OUTPUT_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_BUNDLE_DIR:${CMAKE_TARGET_NAME}>"
            "${MAX_EXTERNAL_OUTPUT_DIR}/${RAW_BASENAME}.${MAX_EXTERN_BUNDLE_EXTENSION}"
            COMMENT "Ensuring ${RAW_BASENAME}.${MAX_EXTERN_BUNDLE_EXTENSION} is in ${MAX_EXTERNAL_OUTPUT_DIR}"
        )
    endif()

    # Add dependency on the shared library copy targets
    # if(TARGET max_copy_torch_libs)
    # add_dependencies(${CMAKE_TARGET_NAME} max_copy_torch_libs)
    # endif()
    # if(TARGET max_copy_contorchionist_core_lib)
    # add_dependencies(${CMAKE_TARGET_NAME} max_copy_contorchionist_core_lib)
    # endif()
    copy_core_libs_to_wrapper_dist(
        TARGET_NAME ${CMAKE_TARGET_NAME}
        DIST_DIR ${MAX_EXTERNAL_OUTPUT_DIR}
    )

    message(STATUS "--- Finished configuring ${RAW_BASENAME} ---")
endfunction()

# --- Automatic Discovery and Creation of Max Externals ---
file(GLOB MAX_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(SOURCE_FILE ${MAX_SOURCE_FILES})
    add_max_external(${SOURCE_FILE})
endforeach()

# --- Common Dependency Copy Targets (REMOVED) ---
# The logic is now handled by the copy_core_libs_to_wrapper_dist function
# and called from within add_max_external.

# --- Help File Copying ---
set(HELP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/help")

if(IS_DIRECTORY "${HELP_SOURCE_DIR}" AND NOT TARGET max_copy_help_files)
    add_custom_target(max_copy_help_files ALL
        COMMENT "Preparing to copy Max help files to ${MAX_EXTERNAL_OUTPUT_DIR}"
    )

    add_custom_command(
        TARGET max_copy_help_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${HELP_SOURCE_DIR}" "${MAX_EXTERNAL_OUTPUT_DIR}"
        COMMENT "Copying Max help files from ${HELP_SOURCE_DIR} to ${MAX_EXTERNAL_OUTPUT_DIR}"
        VERBATIM
    )
    message(STATUS "Configured copy of help directory to ${MAX_EXTERNAL_OUTPUT_DIR}")
endif()

message(STATUS "======== Finished Configuring Max Wrappers ========")