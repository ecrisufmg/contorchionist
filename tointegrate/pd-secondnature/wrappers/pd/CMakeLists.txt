cmake_minimum_required(VERSION 3.18)
project(rmsoverlap_pd_wrapper CXX)

set(PD_DIST_DIR "pd-objects" CACHE STRING "Directory for PD objects in the distribution")

# --- pd.cmake Configuration ---
# Specify PD version, particularly for Windows, to guide pd.cmake's download logic.
# pd.cmake uses this to construct download URLs (e.g., for http://msp.ucsd.edu/Software/pd-${PD_VERSION}.msw.zip).
if(WIN32)
    set(pd_sdk_found FALSE) # Local variable, not cached

    # 1. Check for manually pre-configured PDBINDIR and PD_INCLUDE_DIR
    if(DEFINED PDBINDIR AND IS_DIRECTORY "${PDBINDIR}" AND
       DEFINED PD_INCLUDE_DIR AND IS_DIRECTORY "${PD_INCLUDE_DIR}" AND
       EXISTS "${PDBINDIR}/pd.lib" AND EXISTS "${PD_INCLUDE_DIR}/m_pd.h")
        
        message(STATUS "Attempting to use pre-configured PD SDK paths:")
        message(STATUS "  PDBINDIR: ${PDBINDIR}")
        message(STATUS "  PD_INCLUDE_DIR: ${PD_INCLUDE_DIR}")

        set(PD_ROOT_DIR_CANDIDATE "")
        if(DEFINED PD_ROOT_DIR AND IS_DIRECTORY "${PD_ROOT_DIR}" AND 
           EXISTS "${PD_ROOT_DIR}/bin/pd.lib" AND (EXISTS "${PD_ROOT_DIR}/src/m_pd.h" OR EXISTS "${PD_ROOT_DIR}/include/m_pd.h"))
            file(TO_CMAKE_PATH "${PD_ROOT_DIR}" PD_ROOT_DIR_CANDIDATE)
        else()
            get_filename_component(BIN_PARENT_TEMP "${PDBINDIR}" DIRECTORY)
            get_filename_component(INCLUDE_PARENT_TEMP "${PD_INCLUDE_DIR}" DIRECTORY)
            file(TO_CMAKE_PATH "${BIN_PARENT_TEMP}" NORMALIZED_BIN_PARENT)
            file(TO_CMAKE_PATH "${INCLUDE_PARENT_TEMP}" NORMALIZED_INCLUDE_PARENT)
            file(TO_CMAKE_PATH "${PD_INCLUDE_DIR}" NORMALIZED_PD_INCLUDE_DIR)
            file(TO_CMAKE_PATH "${PDBINDIR}" NORMALIZED_PDBINDIR)

            if("${NORMALIZED_BIN_PARENT}" STREQUAL "${NORMALIZED_INCLUDE_PARENT}")
                set(PD_ROOT_DIR_CANDIDATE "${NORMALIZED_BIN_PARENT}")
            else()
                if(IS_DIRECTORY "${NORMALIZED_BIN_PARENT}/src" AND "${NORMALIZED_PD_INCLUDE_DIR}" STREQUAL "${NORMALIZED_BIN_PARENT}/src")
                    set(PD_ROOT_DIR_CANDIDATE "${NORMALIZED_BIN_PARENT}")
                elseif(IS_DIRECTORY "${NORMALIZED_INCLUDE_PARENT}/bin" AND "${NORMALIZED_PDBINDIR}" STREQUAL "${NORMALIZED_INCLUDE_PARENT}/bin")
                    set(PD_ROOT_DIR_CANDIDATE "${NORMALIZED_INCLUDE_PARENT}")
                endif()
            endif()
        endif()
        
        if(NOT PD_ROOT_DIR_CANDIDATE STREQUAL "")
            set(PD_ROOT_DIR "${PD_ROOT_DIR_CANDIDATE}" CACHE PATH "Root directory of manually provided PD SDK" FORCE)
            message(STATUS "  Using PD_ROOT_DIR: ${PD_ROOT_DIR}")
        else()
            message(WARNING "Could not confidently determine or verify PD_ROOT_DIR from PDBINDIR='${PDBINDIR}' and PD_INCLUDE_DIR='${PD_INCLUDE_DIR}'. This might be okay if paths are correct, or pd.cmake might attempt to find it.")
        endif()

        set(PD_SOURCES_PATH "${PD_INCLUDE_DIR}" CACHE PATH "Path to PD sources/includes (manual)" FORCE)
        set(PD_LIBRARY "${PDBINDIR}/pd.lib" CACHE FILEPATH "Path to pd.lib (manual)" FORCE)
        set(PD_USE_SYSTEM OFF CACHE BOOL "Do not use system-installed Pd (manual override)" FORCE)
        set(PD_DOWNLOAD OFF CACHE BOOL "Do not download Pd (manual override)" FORCE)
        set(pd_sdk_found TRUE)
        
        message(STATUS "Successfully configured PD SDK from manually provided paths.")
        message(STATUS "  PD_SOURCES_PATH: ${PD_SOURCES_PATH}")
        message(STATUS "  PD_LIBRARY: ${PD_LIBRARY}")
    endif()

    # 2. If not found via manual PDBINDIR/PD_INCLUDE_DIR, search for system-installed Pd
    if(NOT pd_sdk_found)
        message(STATUS "Manually specified PDBINDIR/PD_INCLUDE_DIR not found, incomplete, or inconsistent. Searching for system-installed Pure Data...")
        set(PD_SEARCH_PATHS "")
        list(APPEND PD_SEARCH_PATHS "C:/Program Files/Pd" "C:/Program Files (x86)/Pd")

        # Check ENV{ProgramW6432}
        set(ENV_PROGW6432_VAL "$ENV{ProgramW6432}")
        if(ENV_PROGW6432_VAL)
            set(PD_PATH_CANDIDATE "${ENV_PROGW6432_VAL}/Pd")
            if(IS_DIRECTORY "${PD_PATH_CANDIDATE}")
                list(APPEND PD_SEARCH_PATHS "${PD_PATH_CANDIDATE}")
            endif()
        endif()

        # Check ENV{ProgramFiles}
        set(ENV_PROGFILES_VAL "$ENV{ProgramFiles}")
        if(ENV_PROGFILES_VAL)
            set(PD_PATH_CANDIDATE "${ENV_PROGFILES_VAL}/Pd")
            if(IS_DIRECTORY "${PD_PATH_CANDIDATE}")
                list(APPEND PD_SEARCH_PATHS "${PD_PATH_CANDIDATE}")
            endif()
        endif()

        # Check ENV{ProgramFiles(x86)} - using a workaround due to parsing issues with $ENV{ProgramFiles(x86)}
        set(ENV_PROGFILESX86_VAL "") # Initialize
        if(CMAKE_HOST_WIN32)
            execute_process(
                COMMAND cmd /c "echo %ProgramFiles(x86)%"
                OUTPUT_VARIABLE _temp_pf86_cmd_output
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _cmd_result
                ERROR_QUIET
            )
            if(_cmd_result EQUAL 0 AND _temp_pf86_cmd_output)
                if(NOT _temp_pf86_cmd_output STREQUAL "%ProgramFiles(x86)%")
                    file(TO_CMAKE_PATH "${_temp_pf86_cmd_output}" ENV_PROGFILESX86_VAL)
                endif()
            endif()
        endif()

        if(ENV_PROGFILESX86_VAL)
            set(PD_PATH_CANDIDATE "${ENV_PROGFILESX86_VAL}/Pd")
            if(IS_DIRECTORY "${PD_PATH_CANDIDATE}")
                list(APPEND PD_SEARCH_PATHS "${PD_PATH_CANDIDATE}")
            endif()
        endif()

        # Check ENV{PD_DIR}
        set(ENV_PD_DIR_VAL "$ENV{PD_DIR}")
        if(ENV_PD_DIR_VAL AND IS_DIRECTORY "${ENV_PD_DIR_VAL}")
            list(APPEND PD_SEARCH_PATHS "${ENV_PD_DIR_VAL}")
        endif()
        
        if(PD_SEARCH_PATHS) 
            list(REMOVE_DUPLICATES PD_SEARCH_PATHS) 
        endif()
        message(STATUS "Searching for PD in: ${PD_SEARCH_PATHS}")

        set(found_pd_system_install FALSE)
        foreach(SEARCH_PATH_ORIG ${PD_SEARCH_PATHS})
            file(TO_CMAKE_PATH "${SEARCH_PATH_ORIG}" SEARCH_PATH)
            if(NOT IS_DIRECTORY "${SEARCH_PATH}") 
                continue() 
            endif()

            set(TRY_INCLUDE_DIR_SYSTEM_SRC "${SEARCH_PATH}/src")
            set(TRY_INCLUDE_DIR_SYSTEM_INCLUDE "${SEARCH_PATH}/include")
            set(CHOSEN_INCLUDE_DIR_SYSTEM "")

            if(IS_DIRECTORY "${TRY_INCLUDE_DIR_SYSTEM_SRC}" AND EXISTS "${TRY_INCLUDE_DIR_SYSTEM_SRC}/m_pd.h")
                set(CHOSEN_INCLUDE_DIR_SYSTEM "${TRY_INCLUDE_DIR_SYSTEM_SRC}")
            elseif(IS_DIRECTORY "${TRY_INCLUDE_DIR_SYSTEM_INCLUDE}" AND EXISTS "${TRY_INCLUDE_DIR_SYSTEM_INCLUDE}/m_pd.h")
                set(CHOSEN_INCLUDE_DIR_SYSTEM "${TRY_INCLUDE_DIR_SYSTEM_INCLUDE}")
            endif()

            if(IS_DIRECTORY "${SEARCH_PATH}/bin" AND EXISTS "${SEARCH_PATH}/bin/pd.lib" AND NOT CHOSEN_INCLUDE_DIR_SYSTEM STREQUAL "")
                message(STATUS "Found system Pure Data at: ${SEARCH_PATH}")
                set(PD_ROOT_DIR "${SEARCH_PATH}" CACHE PATH "Root directory of system-installed PD SDK" FORCE)
                set(PD_SOURCES_PATH "${CHOSEN_INCLUDE_DIR_SYSTEM}" CACHE PATH "Path to PD sources/includes (system)" FORCE)
                set(PD_INCLUDE_DIR "${CHOSEN_INCLUDE_DIR_SYSTEM}" CACHE PATH "Path to PD include directory (system)" FORCE)
                set(PDBINDIR "${SEARCH_PATH}/bin" CACHE PATH "Path to PD bin directory (system)" FORCE)
                set(PD_LIBRARY "${PDBINDIR}/pd.lib" CACHE FILEPATH "Path to pd.lib (system)" FORCE)
                set(PD_USE_SYSTEM ON CACHE BOOL "Use system-installed Pd" FORCE)
                set(PD_DOWNLOAD OFF CACHE BOOL "Do not download Pd (system found)" FORCE)
                set(pd_sdk_found TRUE)
                set(found_pd_system_install TRUE)
                message(STATUS "Using system-installed PD:")
                message(STATUS "  PD_ROOT_DIR: ${PD_ROOT_DIR}")
                message(STATUS "  PD_SOURCES_PATH: ${PD_SOURCES_PATH}")
                message(STATUS "  PD_INCLUDE_DIR: ${PD_INCLUDE_DIR}")
                message(STATUS "  PDBINDIR: ${PDBINDIR}")
                message(STATUS "  PD_LIBRARY: ${PD_LIBRARY}")
                break()
            endif()
        endforeach()
        if(NOT found_pd_system_install) 
            message(STATUS "No suitable system-installed Pure Data SDK found in common locations.") 
        endif()
    endif()

    # 3. If still not found by manual config or system search, proceed with download logic
    if(NOT pd_sdk_found)
        message(STATUS "Pure Data SDK not found through PDBINDIR/PD_INCLUDE_DIR or system search. Proceeding with download.")
        set(PD_VERSION "0.56-1" CACHE STRING "Pure Data version for Windows (for download)")
        message(STATUS "PD_VERSION for download set to ${PD_VERSION}.")

        # --- Manual PD Download and Extraction for Windows ---
        set(PD_MSW_URL "http://msp.ucsd.edu/Software/pd-${PD_VERSION}.msw.zip")
        # Using CMAKE_BINARY_DIR for paths like build/tmp and build/pd_sdk
        set(PD_DOWNLOAD_TARGET_DIR "${CMAKE_BINARY_DIR}/tmp")
        set(PD_ZIP_FILE "${PD_DOWNLOAD_TARGET_DIR}/pd-${PD_VERSION}.msw.zip")
        set(PD_EXTRACT_TARGET_DIR "${CMAKE_BINARY_DIR}/pd_windows_binary")

        # Ensure target directories exist
        file(MAKE_DIRECTORY ${PD_DOWNLOAD_TARGET_DIR})
        file(MAKE_DIRECTORY ${PD_EXTRACT_TARGET_DIR})

        # Download PD only if the ZIP file doesn't exist
        if(NOT EXISTS "${PD_ZIP_FILE}")
            message(STATUS "Downloading Pure Data from ${PD_MSW_URL} to ${PD_ZIP_FILE}")
            file(DOWNLOAD ${PD_MSW_URL} ${PD_ZIP_FILE}
                 SHOW_PROGRESS
                 STATUS download_status
                 TIMEOUT 300 # 5 minutes timeout
            )
            list(GET download_status 0 download_error_code)
            if(NOT download_error_code EQUAL 0)
                message(FATAL_ERROR "Failed to download Pure Data: ${download_status}")
            endif()
        else()
            message(STATUS "Pure Data ZIP already downloaded: ${PD_ZIP_FILE}")
        endif()

        # Extract PD only if not already extracted (check for a key directory)
        # The zip typically contains a top-level folder like "pd-0.55-2"
        set(PD_EXPECTED_EXTRACTED_FOLDER "${PD_EXTRACT_TARGET_DIR}/pd-${PD_VERSION}")
        set(PD_EXTRACT_CHECK_DIR "${PD_EXPECTED_EXTRACTED_FOLDER}/src") # Check for the 'src' dir

        if(NOT EXISTS "${PD_EXTRACT_CHECK_DIR}")
            message(STATUS "Extracting ${PD_ZIP_FILE} to ${PD_EXTRACT_TARGET_DIR}")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xf "${PD_ZIP_FILE}"
                WORKING_DIRECTORY ${PD_EXTRACT_TARGET_DIR} # Extracts into this directory
                RESULT_VARIABLE extract_result
                OUTPUT_VARIABLE extract_output # Capture output for debugging
                ERROR_VARIABLE extract_error   # Capture error for debugging
            )
            if(NOT extract_result EQUAL 0)
                message(FATAL_ERROR "Failed to extract Pure Data ZIP. Result: ${extract_result}\nOutput: ${extract_output}\nError: ${extract_error}")
            else()
                message(STATUS "Pure Data extracted successfully to ${PD_EXTRACT_TARGET_DIR}")
            endif()
        else()
            message(STATUS "Pure Data already extracted in ${PD_EXPECTED_EXTRACTED_FOLDER}")
        endif()

        # Set PD_INCLUDE_DIR and PDBINDIR for pd.cmake to use this version
        # These paths point into the extracted "pd-${PD_VERSION}" folder
        if(EXISTS "${PD_EXPECTED_EXTRACTED_FOLDER}/src" AND EXISTS "${PD_EXPECTED_EXTRACTED_FOLDER}/bin")
            set(PD_ROOT_DIR "${PD_EXPECTED_EXTRACTED_FOLDER}" CACHE PATH "Root directory of manually provided PD SDK" FORCE)
            set(PD_SOURCES_PATH "${PD_EXPECTED_EXTRACTED_FOLDER}/src" CACHE PATH "Path to PD sources (used by pd.cmake)" FORCE)
            set(PD_INCLUDE_DIR "${PD_EXPECTED_EXTRACTED_FOLDER}/src" CACHE PATH "Path to PD include directory" FORCE)
            set(PDBINDIR "${PD_EXPECTED_EXTRACTED_FOLDER}/bin" CACHE PATH "Path to PD bin directory (used by pd.cmake)" FORCE)
            set(PD_LIBRARY "${PDBINDIR}/pd.lib" CACHE FILEPATH "Path to pd.lib" FORCE)

            # Explicitly tell pd.cmake not to search system or download again
            set(PD_USE_SYSTEM OFF CACHE BOOL "Do not use system-installed Pd" FORCE)
            set(PD_DOWNLOAD OFF CACHE BOOL "Do not download Pd" FORCE)

            message(STATUS "Using manually downloaded/extracted PD:")
            message(STATUS "  PD_ROOT_DIR: ${PD_ROOT_DIR}")
            message(STATUS "  PD_SOURCES_PATH: ${PD_SOURCES_PATH}")
            message(STATUS "  PD_INCLUDE_DIR: ${PD_INCLUDE_DIR}")
            message(STATUS "  PDBINDIR: ${PDBINDIR}")
            message(STATUS "  PD_LIBRARY: ${PD_LIBRARY}")
            message(STATUS "  PD_USE_SYSTEM: ${PD_USE_SYSTEM}")
            message(STATUS "  PD_DOWNLOAD: ${PD_DOWNLOAD}")
        else()
            message(WARNING "Manually downloaded PD SDK not found at expected location: ${PD_EXPECTED_EXTRACTED_FOLDER}. pd.cmake might try to download it again or fail.")
        endif()
        # --- End of Manual PD Download and Extraction ---
    else() # This 'else' corresponds to 'if(NOT pd_sdk_found)'
        message(STATUS "Pure Data SDK was found either through manual configuration or system search. Skipping download.")
    endif() # End of 'if(NOT pd_sdk_found)' / 'else' for download skip
endif() # End of if(WIN32)

# Set custom external suffixes for different platforms.
# These will be used by pd_add_external if set before including pd.cmake.
# This overrides pd.cmake's default suffixes if needed.
# if(WIN32)
#     set(PD_EXTERNAL_SUFFIX ".m_amd64" CACHE STRING "PD external suffix for Windows")
# elseif(APPLE)
#     # Using suffix from your previous configuration
#     set(PD_EXTERNAL_SUFFIX ".darwin-fat-32.so" CACHE STRING "PD external suffix for macOS")
# elseif(UNIX AND NOT APPLE) # Linux
#     if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
#         set(PD_EXTERNAL_SUFFIX ".l_amd64" CACHE STRING "PD external suffix for Linux x86_64")
#     else() # Assuming arm64 or other Linux architecture
#         set(PD_EXTERNAL_SUFFIX ".l_arm64" CACHE STRING "PD external suffix for Linux arm64")
#     endif()
# endif()

if(DEFINED PD_EXTERNAL_SUFFIX)
    message(STATUS "Using custom PD_EXTERNAL_SUFFIX: ${PD_EXTERNAL_SUFFIX}")
else()
    message(STATUS "Using pd.cmake default PD_EXTERNAL_SUFFIX for this platform.")
endif()

# Set the path to the pd.cmake directory.
# Assumes pd.cmake submodule is at medusa/thirdparty/pd.cmake,
# and this CMakeLists.txt is in medusa/wrappers/pd/

# Ensure PD_USE_SYSTEM and PD_DOWNLOAD are set to OFF if not on Windows,
# to prevent pd.cmake from downloading if not desired (though our manual download is Windows-specific)
if(NOT WIN32)
    # For non-Windows, you might want to control this differently,
    # but if manual setup is preferred everywhere, these would be OFF.
    # For now, let's assume system PD is fine or handled by pd.cmake's defaults on other platforms.
    # If you had manual setup for other platforms, you'd set these to OFF there too.
endif()

set(PDCMAKE_DIR ../../thirdparty/pd.cmake CACHE PATH "Path to pd.cmake directory")
include(${PDCMAKE_DIR}/pd.cmake) # This was line 121 in your error log

message(STATUS "DEBUG: PD_LIBRARY after include(pd.cmake) = '${PD_LIBRARY}'")
message(STATUS "DEBUG: PD_INCLUDE_DIR after include(pd.cmake) = '${PD_INCLUDE_DIR}'")

# --- Glob for Source Files ---
file(GLOB PD_EXTERNAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Define the path to the pd_arg_parser utility
set(PD_ARG_PARSER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/pd_arg_parser.cpp")
set(PD_ARG_PARSER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils/include")

message(STATUS "Found PD external sources: ${PD_EXTERNAL_SOURCES}")
message(STATUS "PD Arg Parser source: ${PD_ARG_PARSER_SRC}")
message(STATUS "PD Arg Parser include dir: ${PD_ARG_PARSER_INCLUDE_DIR}")

# --- Loop Through Sources and Build Each External ---
foreach(SOURCE_FILE ${PD_EXTERNAL_SOURCES})
    get_filename_component(PD_EXT_FILENAME ${SOURCE_FILE} NAME) # e.g., rmsoverlap~.cpp
    string(REGEX REPLACE "\\.cpp$" "" PD_EXT_NAME ${PD_EXT_FILENAME}) # e.g., rmsoverlap~ (this is the name PD uses)
    
    # Create a CMake-safe target name
    string(REPLACE "~" "_tilde" CMAKE_TARGET_NAME_BASE ${PD_EXT_NAME})
    set(CMAKE_TARGET_NAME "pd_${CMAKE_TARGET_NAME_BASE}") # e.g., pd_rmsoverlap_tilde to ensure uniqueness

    message(STATUS "Processing PD external: ${PD_EXT_NAME} (Source: ${SOURCE_FILE}, CMake Target: ${CMAKE_TARGET_NAME})")

    # Add the external using the PD name and the CMake target name
    # This assumes your pd.cmake supports the TARGET keyword.
    pd_add_external(${PD_EXT_NAME} ${SOURCE_FILE} TARGET ${CMAKE_TARGET_NAME})

    # Explicitly add the pd_arg_parser.cpp to the target sources
    target_sources(${CMAKE_TARGET_NAME} PRIVATE ${PD_ARG_PARSER_SRC})

    # Set the output name for the target to ensure correct filename (with ~)
    set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES 
        OUTPUT_NAME ${PD_EXT_NAME}
    )

    # Add include directories for the shared core and the pd_arg_parser utility
    target_include_directories(${CMAKE_TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../../shared_core/include"
        ${PD_ARG_PARSER_INCLUDE_DIR}
    )

    # Link against the shared core library
    if(TARGET SharedCore)
        target_link_libraries(${CMAKE_TARGET_NAME} PRIVATE SharedCore)
    else()
        message(WARNING "Target SharedCore does not exist. Cannot link ${CMAKE_TARGET_NAME}.")
    endif()

    # Set output directories
    # pd.cmake should use PD_EXTERNAL_SUFFIX for the extension.
    # OUTPUT_NAME property (set above) ensures the base name is correct.
    set_target_properties(${CMAKE_TARGET_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist/${PD_DIST_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist/${PD_DIST_DIR}"
    )
    
    # --- Help File Installation ---
    set(HELP_FILE_BASENAME "${PD_EXT_NAME}-help.pd")
    set(HELP_FILE_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/help/${HELP_FILE_BASENAME}")
    set(HELP_FILE_DEST_DIR "$<TARGET_FILE_DIR:${CMAKE_TARGET_NAME}>") # Destination is where the .pd_linux/.dll is

    if(EXISTS "${HELP_FILE_SOURCE_PATH}")
        add_custom_command(
            TARGET ${CMAKE_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                    "${HELP_FILE_SOURCE_PATH}"
                    "${HELP_FILE_DEST_DIR}/${HELP_FILE_BASENAME}"
            COMMENT "Copying help file ${HELP_FILE_BASENAME} for ${PD_EXT_NAME} to ${HELP_FILE_DEST_DIR}"
        )
        message(STATUS "Configured help file copy for ${PD_EXT_NAME}: ${HELP_FILE_SOURCE_PATH} -> ${HELP_FILE_DEST_DIR}/${HELP_FILE_BASENAME}")
    else()
        message(WARNING "Help file not found for ${PD_EXT_NAME}: ${HELP_FILE_SOURCE_PATH}")
    endif()
endforeach()

# Note: pd.cmake typically handles installation rules if its variable PD_INSTALL_LIBS is ON (default).
