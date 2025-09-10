cmake_minimum_required(VERSION 3.15)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch - use the same configuration as the main project
find_package(Torch REQUIRED)

# TORCH_LIBRARIES should be available from the parent scope (root CMakeLists.txt)
# If not, you might need to ensure it's properly set or re-find it here.

# The contorchionist_core target should be available.

# Helper function to create test executables with GTest dependencies
function(create_gtest_executable target_name source_file)
    find_package(GTest REQUIRED)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE
        contorchionist_core # Link against the core library
        ${TORCH_LIBRARIES} # Link against Torch libraries
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
    )

    # Set platform-specific RPATH
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BIN_DIR}" # Output to build/dist/bin
            INSTALL_RPATH "@loader_path" # RPATH to find libs in the executable's own directory (macOS)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    else()
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BIN_DIR}" # Output to build/dist/bin
            INSTALL_RPATH "$ORIGIN" # RPATH to find libs in the executable's own directory (Linux)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    endif()

    add_test(NAME ${target_name} COMMAND ${target_name} WORKING_DIRECTORY "${PROJECT_BIN_DIR}")

    # --- Copy contorchionist_core to executable's directory ---
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_LIB_DIR}/$<TARGET_FILE_NAME:contorchionist_core>" # Source from build/dist/lib
        "$<TARGET_FILE_DIR:${target_name}>/$<TARGET_FILE_NAME:contorchionist_core>" # Destination: executable's dir
        COMMENT "Copying contorchionist_core for ${target_name} to executable's directory"
    )

    # --- Copy Torch dylibs to executable's directory ---
    if(APPLE AND MACOS_TORCH_DYLIBS_FILENAMES AND Torch_INSTALL_PREFIX)
        foreach(dylib_filename ${MACOS_TORCH_DYLIBS_FILENAMES})
            set(source_dylib_path "${Torch_INSTALL_PREFIX}/lib/${dylib_filename}")

            if(EXISTS ${source_dylib_path})
                add_custom_command(
                    TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${source_dylib_path}"
                    "$<TARGET_FILE_DIR:${target_name}>/${dylib_filename}" # Destination: executable's dir
                    COMMENT "Copying Torch dylib ${dylib_filename} for ${target_name} to executable's directory"
                )
            endif()
        endforeach()
    endif()

    # --- Copy GTest libraries to executable's directory ---
    if(GTest_FOUND)
        if(APPLE)
            # For GTest::gtest
            get_target_property(GTEST_LIBRARY GTest::gtest LOCATION)

            if(GTEST_LIBRARY AND EXISTS ${GTEST_LIBRARY} AND IS_ABSOLUTE ${GTEST_LIBRARY})
                get_filename_component(GTEST_DYLIB_NAME ${GTEST_LIBRARY} NAME)
                add_custom_command(
                    TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GTEST_LIBRARY}"
                    "$<TARGET_FILE_DIR:${target_name}>/${GTEST_DYLIB_NAME}"
                    COMMENT "Copying GTest dylib ${GTEST_DYLIB_NAME} for ${target_name} to executable's directory"
                )
            endif()

            # For GTest::gmock
            get_target_property(GMOCK_LIBRARY GTest::gmock LOCATION)

            if(GMOCK_LIBRARY AND EXISTS ${GMOCK_LIBRARY} AND IS_ABSOLUTE ${GMOCK_LIBRARY})
                get_filename_component(GMOCK_DYLIB_NAME ${GMOCK_LIBRARY} NAME)
                add_custom_command(
                    TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GMOCK_LIBRARY}"
                    "$<TARGET_FILE_DIR:${target_name}>/${GMOCK_DYLIB_NAME}"
                    COMMENT "Copying GTest dylib ${GMOCK_DYLIB_NAME} for ${target_name} to executable's directory"
                )
            endif()

            # For GTest::gtest_main (usually a static library, but check just in case)
            get_target_property(GTEST_MAIN_LIBRARY GTest::gtest_main LOCATION)

            if(GTEST_MAIN_LIBRARY AND EXISTS ${GTEST_MAIN_LIBRARY} AND IS_ABSOLUTE ${GTEST_MAIN_LIBRARY})
                get_filename_component(GTEST_MAIN_DYLIB_NAME ${GTEST_MAIN_LIBRARY} NAME)

                # Check if it's actually a dylib before trying to copy
                if(GTEST_MAIN_DYLIB_NAME MATCHES "\\.dylib$")
                    add_custom_command(
                        TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GTEST_MAIN_LIBRARY}"
                        "$<TARGET_FILE_DIR:${target_name}>/${GTEST_MAIN_DYLIB_NAME}"
                        COMMENT "Copying GTest dylib ${GTEST_MAIN_DYLIB_NAME} for ${target_name} to executable's directory"
                    )
                endif()
            endif()
        else()
            # Linux: Copy GTest shared libraries
            # For GTest::gtest
            get_target_property(GTEST_LIBRARY GTest::gtest LOCATION)

            if(GTEST_LIBRARY AND EXISTS ${GTEST_LIBRARY} AND IS_ABSOLUTE ${GTEST_LIBRARY})
                get_filename_component(GTEST_SO_NAME ${GTEST_LIBRARY} NAME)

                if(GTEST_SO_NAME MATCHES "\\.so")
                    add_custom_command(
                        TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GTEST_LIBRARY}"
                        "$<TARGET_FILE_DIR:${target_name}>/${GTEST_SO_NAME}"
                        COMMENT "Copying GTest so ${GTEST_SO_NAME} for ${target_name} to executable's directory"
                    )
                endif()
            endif()

            # For GTest::gmock
            get_target_property(GMOCK_LIBRARY GTest::gmock LOCATION)

            if(GMOCK_LIBRARY AND EXISTS ${GMOCK_LIBRARY} AND IS_ABSOLUTE ${GMOCK_LIBRARY})
                get_filename_component(GMOCK_SO_NAME ${GMOCK_LIBRARY} NAME)

                if(GMOCK_SO_NAME MATCHES "\\.so")
                    add_custom_command(
                        TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GMOCK_LIBRARY}"
                        "$<TARGET_FILE_DIR:${target_name}>/${GMOCK_SO_NAME}"
                        COMMENT "Copying GTest so ${GMOCK_SO_NAME} for ${target_name} to executable's directory"
                    )
                endif()
            endif()

            # For GTest::gtest_main
            get_target_property(GTEST_MAIN_LIBRARY GTest::gtest_main LOCATION)

            if(GTEST_MAIN_LIBRARY AND EXISTS ${GTEST_MAIN_LIBRARY} AND IS_ABSOLUTE ${GTEST_MAIN_LIBRARY})
                get_filename_component(GTEST_MAIN_SO_NAME ${GTEST_MAIN_LIBRARY} NAME)

                if(GTEST_MAIN_SO_NAME MATCHES "\\.so")
                    add_custom_command(
                        TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GTEST_MAIN_LIBRARY}"
                        "$<TARGET_FILE_DIR:${target_name}>/${GTEST_MAIN_SO_NAME}"
                        COMMENT "Copying GTest so ${GTEST_MAIN_SO_NAME} for ${target_name} to executable's directory"
                    )
                endif()
            endif()
        endif()
    endif()
endfunction()

# Helper function to create standalone test executables (without GTest)
function(create_standalone_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE
        contorchionist_core # Link against the core library
        ${TORCH_LIBRARIES} # Link against Torch libraries
    )

    # Set platform-specific RPATH
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BIN_DIR}" # Output to build/dist/bin
            INSTALL_RPATH "@loader_path" # RPATH to find libs in the executable's own directory (macOS)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    else()
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BIN_DIR}" # Output to build/dist/bin
            INSTALL_RPATH "$ORIGIN" # RPATH to find libs in the executable's own directory (Linux)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    endif()

    add_test(NAME ${target_name} COMMAND ${target_name} WORKING_DIRECTORY "${PROJECT_BIN_DIR}")

    # --- Copy contorchionist_core to executable's directory ---
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_LIB_DIR}/$<TARGET_FILE_NAME:contorchionist_core>" # Source from build/dist/lib
        "$<TARGET_FILE_DIR:${target_name}>/$<TARGET_FILE_NAME:contorchionist_core>" # Destination: executable's dir
        COMMENT "Copying contorchionist_core for ${target_name} to executable's directory"
    )

    # --- Copy Torch dylibs to executable's directory ---
    if(APPLE AND MACOS_TORCH_DYLIBS_FILENAMES AND Torch_INSTALL_PREFIX)
        foreach(dylib_filename ${MACOS_TORCH_DYLIBS_FILENAMES})
            set(source_dylib_path "${Torch_INSTALL_PREFIX}/lib/${dylib_filename}")

            if(EXISTS ${source_dylib_path})
                add_custom_command(
                    TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${source_dylib_path}"
                    "$<TARGET_FILE_DIR:${target_name}>/${dylib_filename}" # Destination: executable's dir
                    COMMENT "Copying Torch dylib ${dylib_filename} for ${target_name} to executable's directory"
                )
            endif()
        endforeach()
    endif()
endfunction()

# Create test executables
create_gtest_executable(test_circular_buffer_complete test_circular_buffer_complete.cpp)
create_gtest_executable(test_device_utils test_device_utils.cpp)
create_gtest_executable(test_neural_registry test_neural_registry.cpp)
create_gtest_executable(test_activation_registry test_activation_registry.cpp)
create_gtest_executable(test_neural_layers test_neural_layers.cpp)

# create_gtest_executable(test_model_manager test_model_manager.cpp)
create_gtest_executable(test_tensor_utils test_tensor_utils.cpp)
create_standalone_executable(test_unit_conversions test_unit_conversions.cpp)
create_gtest_executable(test_integration test_integration.cpp)
create_standalone_executable(test_audio_features test_audio_features.cpp)
create_standalone_executable(test_a_dct test_a_dct.cpp)

# create_standalone_executable(test_mel_filterbank test_mel_filterbank.cpp) # Replaced by test_a_melspectrogram
create_gtest_executable(test_a_melspectrogram test_a_melspectrogram.cpp) # New test for MelSpectrogramProcessor
create_standalone_executable(test_melspectrogram_comparison test_melspectrogram_comparison.cpp) # Comparison with Librosa reference
create_standalone_executable(test_cuda_minimal test_cuda_minimal.cpp)
create_standalone_executable(test_melspectrogram_pipeline test_melspectrogram_pipeline.cpp)
create_standalone_executable(test_a_spectrogram test_a_spectrogram.cpp)
create_gtest_executable(test_normalizations test_normalizations.cpp)

# Custom target to run all tests (optional, CTest is preferred)
# ... (your existing add_custom_target for run_all_tests can remain if you use it)
# Consider using 'ctest' command from the build directory instead.
