cmake_minimum_required(VERSION 3.18)
project(contorchionist_sc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- SuperCollider SDK Discovery (adapted from medusa example) ---
set(SC_INCLUDE_DIRS_FINAL "")
set(SuperCollider_FOUND FALSE)

find_package(SuperCollider QUIET)
if(SuperCollider_FOUND AND DEFINED SuperCollider_INCLUDE_DIRS)
    message(STATUS "Found SuperCollider SDK via find_package: ${SuperCollider_INCLUDE_DIRS}")
    set(SC_INCLUDE_DIRS_FINAL ${SuperCollider_INCLUDE_DIRS})
else()
    if(SuperCollider_FOUND)
        message(WARNING "find_package(SuperCollider) succeeded but SuperCollider_INCLUDE_DIRS is not set. Treating as not found.")
        set(SuperCollider_FOUND FALSE)
    endif()

    set(SC_PATH_FROM_ENV "$ENV{SC_PATH}")
    if(DEFINED ENV{SC_PATH} AND NOT SC_PATH_FROM_ENV STREQUAL "")
        message(STATUS "Checking for SuperCollider SDK in ENV{SC_PATH}: ${SC_PATH_FROM_ENV}")
        set(TRY_SC_PLUGIN_DIR "${SC_PATH_FROM_ENV}/include/plugin_interface")
        set(TRY_SC_COMMON_DIR "${SC_PATH_FROM_ENV}/include/common")
        if(EXISTS "${TRY_SC_PLUGIN_DIR}/SC_PlugIn.h" AND EXISTS "${TRY_SC_COMMON_DIR}/SC_Types.h")
            message(STATUS "Found SuperCollider headers via SC_PATH: ${TRY_SC_PLUGIN_DIR} and ${TRY_SC_COMMON_DIR}")
            set(SC_INCLUDE_DIRS_FINAL "${TRY_SC_PLUGIN_DIR}" "${TRY_SC_COMMON_DIR}")
            set(SuperCollider_FOUND TRUE)
        endif()
    endif()

    if(NOT SuperCollider_FOUND)
        set(SC_CLONE_PARENT_DIR "${CMAKE_BINARY_DIR}/sc_sdk_clone")
        set(SC_CLONE_DIR "${SC_CLONE_PARENT_DIR}/supercollider")
        set(SC_EXPECTED_PLUGIN_HEADER "${SC_CLONE_DIR}/include/plugin_interface/SC_PlugIn.h")
        set(SC_EXPECTED_COMMON_HEADER "${SC_CLONE_DIR}/include/common/SC_Types.h")

        if(EXISTS "${SC_EXPECTED_PLUGIN_HEADER}" AND EXISTS "${SC_EXPECTED_COMMON_HEADER}")
            message(STATUS "Found existing valid SuperCollider clone at ${SC_CLONE_DIR}.")
            set(SC_INCLUDE_DIRS_FINAL "${SC_CLONE_DIR}/include/plugin_interface" "${SC_CLONE_DIR}/include/common")
            set(SuperCollider_FOUND TRUE)
        else()
            message(STATUS "SuperCollider SDK not found. Attempting to clone from GitHub.")
            find_package(Git QUIET)
            if(GIT_FOUND)
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} clone --depth 1 https://github.com/supercollider/supercollider.git "${SC_CLONE_DIR}"
                    RESULT_VARIABLE git_clone_result
                )
                if(git_clone_result EQUAL 0 AND EXISTS "${SC_EXPECTED_PLUGIN_HEADER}")
                    message(STATUS "SuperCollider cloned successfully.")
                    set(SC_INCLUDE_DIRS_FINAL "${SC_CLONE_DIR}/include/plugin_interface" "${SC_CLONE_DIR}/include/common")
                    set(SuperCollider_FOUND TRUE)
                else()
                    message(WARNING "Failed to clone SuperCollider repository.")
                endif()
            else()
                message(WARNING "Git not found. Cannot clone SuperCollider.")
            endif()
        endif()
    endif()
endif()

if(NOT SuperCollider_FOUND OR SC_INCLUDE_DIRS_FINAL STREQUAL "")
    message(FATAL_ERROR "Could not determine SuperCollider include directories. "
                        "Please ensure SuperCollider SDK is installed and discoverable by find_package(SuperCollider), "
                        "or set the SC_PATH environment variable.")
endif()

# --- RPATH settings for portability ---
if(APPLE)
    set(_rpath "@loader_path")
elseif(UNIX)
    set(_rpath "$ORIGIN")
else()
    set(_rpath "")
endif()

# --- Output Directory ---
set(SC_EXTERNAL_OUTPUT_DIR "${PROJECT_DIST_DIR}/supercollider")
set(SC_CLASSES_OUTPUT_DIR "${SC_EXTERNAL_OUTPUT_DIR}/Classes")
file(MAKE_DIRECTORY ${SC_EXTERNAL_OUTPUT_DIR})
file(MAKE_DIRECTORY ${SC_CLASSES_OUTPUT_DIR})

# --- Glob for Source Files ---
file(GLOB SC_UGEN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# --- Loop Through Sources and Build Each UGen ---
foreach(SOURCE_FILE ${SC_UGEN_SOURCES})
    get_filename_component(UGEN_FILENAME ${SOURCE_FILE} NAME_WE)
    set(CMAKE_TARGET_NAME "sc_${UGEN_FILENAME}")

    message(STATUS "Configuring SC UGen: ${UGEN_FILENAME}")

    add_library(${CMAKE_TARGET_NAME} MODULE ${SOURCE_FILE})

    target_include_directories(${CMAKE_TARGET_NAME} PRIVATE
        ${SC_INCLUDE_DIRS_FINAL}
        "${CMAKE_SOURCE_DIR}/core/include"
    )

    if(TARGET contorchionist::core)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE contorchionist::core)
    else()
        message(WARNING "Target 'contorchionist::core' not found for linking.")
    endif()

    if(WIN32)
        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES PREFIX "" SUFFIX ".scx" OUTPUT_NAME "${UGEN_FILENAME}" LIBRARY_OUTPUT_DIRECTORY "${SC_EXTERNAL_OUTPUT_DIR}")
    elseif(APPLE)
        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES PREFIX "" SUFFIX ".scx" OUTPUT_NAME "${UGEN_FILENAME}" LIBRARY_OUTPUT_DIRECTORY "${SC_EXTERNAL_OUTPUT_DIR}")
    else() # Linux
        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES OUTPUT_NAME "${UGEN_FILENAME}" LIBRARY_OUTPUT_DIRECTORY "${SC_EXTERNAL_OUTPUT_DIR}")
    endif()

    # Set RPATH for non-Windows platforms to allow finding co-located .dylibs/.sos
    if(NOT WIN32)
        set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
            INSTALL_RPATH "${_rpath}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    copy_core_libs_to_wrapper_dist(
        TARGET_NAME ${CMAKE_TARGET_NAME}
        DIST_DIR ${SC_EXTERNAL_OUTPUT_DIR}
    )

    # --- Copy .sc class definition file ---
    set(SC_CLASS_FILE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/${UGEN_FILENAME}.sc")
    if(EXISTS "${SC_CLASS_FILE_SOURCE}")
        add_custom_command(
            TARGET ${CMAKE_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${SC_CLASS_FILE_SOURCE}"
                "${SC_CLASSES_OUTPUT_DIR}/${UGEN_FILENAME}.sc"
            COMMENT "Copying SC class definition ${UGEN_FILENAME}.sc to ${SC_CLASSES_OUTPUT_DIR}"
        )
    endif()
endforeach()

# --- Copy Help Files ---
set(HELP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schelp")
set(HELP_DEST_DIR "${SC_EXTERNAL_OUTPUT_DIR}/HelpSource/Classes")
if(IS_DIRECTORY "${HELP_SOURCE_DIR}")
    add_custom_target(copy_sc_help ALL COMMENT "Copying SuperCollider help files")
    add_custom_command(
        TARGET copy_sc_help POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HELP_DEST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${HELP_SOURCE_DIR}/" "${HELP_DEST_DIR}/"
        COMMENT "Copying SC help files to ${HELP_DEST_DIR}"
    )
endif()

message(STATUS "SuperCollider UGen wrapper build configuration complete.")
