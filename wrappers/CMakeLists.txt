# CMakeLists.txt for src/wrappers

# This CMakeLists.txt can be used to conditionally add the subdirectories
# for each wrapper based on the options defined in the main CMakeLists.txt

# --- Helper function to copy core libraries to a wrapper's distribution directory ---
function(copy_core_libs_to_wrapper_dist)
    set(options)
    set(oneValueArgs TARGET_NAME DIST_DIR)
    set(multiValueArgs)
    cmake_parse_arguments(COPY_LIBS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT COPY_LIBS_TARGET_NAME OR NOT COPY_LIBS_DIST_DIR)
        message(FATAL_ERROR "copy_core_libs_to_wrapper_dist requires TARGET_NAME and DIST_DIR.")
    endif()

    # Copy contorchionist_core library
    add_custom_command(
        TARGET ${COPY_LIBS_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:contorchionist_core>
        ${COPY_LIBS_DIST_DIR}
        COMMENT "Copying contorchionist_core shared library to ${COPY_LIBS_DIST_DIR}"
    )

    # Copy LibTorch libraries
    if(APPLE AND MACOS_TORCH_DYLIBS_FILENAMES)
        foreach(dylib_filename ${MACOS_TORCH_DYLIBS_FILENAMES})
            set(source_dylib_path "${Torch_INSTALL_PREFIX}/lib/${dylib_filename}")
            if(EXISTS ${source_dylib_path})
                add_custom_command(
                    TARGET ${COPY_LIBS_TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${source_dylib_path}"
                    "${COPY_LIBS_DIST_DIR}/${dylib_filename}"
                    COMMENT "Copying ${dylib_filename} to ${COPY_LIBS_DIST_DIR}"
                )
            endif()
        endforeach()
    elseif(UNIX AND NOT APPLE AND LINUX_TORCH_SOS_FILENAMES)
        # Handle Linux .so files
    elseif(WIN32)
        # Handle Windows .dll files
    endif()
endfunction()


if(BUILD_PD_WRAPPER)
    add_subdirectory(puredata)
endif()

if(BUILD_MAX_WRAPPER)
    add_subdirectory(max)
endif()

if(BUILD_PYTHON_WRAPPER)
    add_subdirectory(python)
endif()

if(BUILD_SC_WRAPPER)
    add_subdirectory(sc)
endif()
