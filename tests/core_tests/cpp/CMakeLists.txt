cmake_minimum_required(VERSION 3.15)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch - use the same configuration as the main project
find_package(Torch REQUIRED)
# TORCH_LIBRARIES should be available from the parent scope (root CMakeLists.txt)
# If not, you might need to ensure it's properly set or re-find it here.

# The contorchionist_core target should be available.

# Create a single target to copy all shared libraries once
add_custom_target(copy_test_libraries
    COMMENT "Copying shared libraries for tests"
)

# Add commands to copy contorchionist_core
add_custom_command(
    TARGET copy_test_libraries PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_LIB_DIR}/$<TARGET_FILE_NAME:contorchionist_core>"
        "${PROJECT_TESTS_DIR}/$<TARGET_FILE_NAME:contorchionist_core>"
    COMMENT "Copying contorchionist_core to test directory"
)

# Add commands to copy Torch dylibs (cross-platform)
if(TORCH_LIBRARIES)
    foreach(torch_lib ${TORCH_LIBRARIES})
        # Check if it's a shared library file (dylib, so, dll)
        if(torch_lib MATCHES "\\.(dylib|so|dll)$")
            get_filename_component(lib_filename ${torch_lib} NAME)
            add_custom_command(
                TARGET copy_test_libraries PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${torch_lib}"
                    "${PROJECT_TESTS_DIR}/${lib_filename}"
                COMMENT "Copying Torch library ${lib_filename} to test directory"
            )
        elseif(IS_ABSOLUTE ${torch_lib} AND EXISTS ${torch_lib})
            # Handle cases where the library path doesn't have extension but exists
            get_filename_component(lib_filename ${torch_lib} NAME)
            add_custom_command(
                TARGET copy_test_libraries PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${torch_lib}"
                    "${PROJECT_TESTS_DIR}/${lib_filename}"
                COMMENT "Copying Torch library ${lib_filename} to test directory"
            )
        endif()
    endforeach()
endif()

# Fallback: Use platform-specific dylib detection if MACOS_TORCH_DYLIBS_FILENAMES is available
if(MACOS_TORCH_DYLIBS_FILENAMES AND Torch_INSTALL_PREFIX)
    foreach(dylib_filename ${MACOS_TORCH_DYLIBS_FILENAMES})
        set(source_dylib_path "${Torch_INSTALL_PREFIX}/lib/${dylib_filename}")
        if(EXISTS ${source_dylib_path})
            add_custom_command(
                TARGET copy_test_libraries PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${source_dylib_path}"
                    "${PROJECT_TESTS_DIR}/${dylib_filename}"
                COMMENT "Copying Torch dylib ${dylib_filename} to test directory"
            )
        endif()
    endforeach()
endif()

# Add commands to copy GTest libraries (cross-platform)
if(GTest_FOUND)
    # List of GTest targets to check
    set(GTEST_TARGETS GTest::gtest GTest::gmock GTest::gtest_main)
    
    foreach(gtest_target ${GTEST_TARGETS})
        if(TARGET ${gtest_target})
            get_target_property(GTEST_LIBRARY ${gtest_target} LOCATION)
            if(GTEST_LIBRARY AND EXISTS ${GTEST_LIBRARY} AND IS_ABSOLUTE ${GTEST_LIBRARY})
                get_filename_component(GTEST_LIB_NAME ${GTEST_LIBRARY} NAME)
                # Only copy if it's a shared library (dylib, so, dll)
                if(GTEST_LIB_NAME MATCHES "\\.(dylib|so|dll)$")
                    add_custom_command(
                        TARGET copy_test_libraries PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${GTEST_LIBRARY}"
                            "${PROJECT_TESTS_DIR}/${GTEST_LIB_NAME}"
                        COMMENT "Copying GTest library ${GTEST_LIB_NAME} to test directory"
                    )
                endif()
            endif()
        endif()
    endforeach()
endif()

# Make copy_test_libraries depend on contorchionist_core being built
add_dependencies(copy_test_libraries contorchionist_core)

# Helper function to create test executables with GTest dependencies
function(create_gtest_executable target_name source_file)
    find_package(GTest REQUIRED)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE
        contorchionist_core # Link against the core library
        ${TORCH_LIBRARIES}  # Link against Torch libraries
        GTest::gmock
        GTest::gtest_main  # gtest_main already includes gtest
    )
    # Set platform-specific RPATH
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_TESTS_DIR}" # Output to build/dist/tests
            INSTALL_RPATH "@loader_path" # RPATH to find libs in the executable's own directory (macOS)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    else()
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_TESTS_DIR}" # Output to build/dist/tests
            INSTALL_RPATH "$ORIGIN" # RPATH to find libs in the executable's own directory (Linux)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name} WORKING_DIRECTORY "${PROJECT_TESTS_DIR}")

    # Make this test depend on the shared library copy target
    add_dependencies(${target_name} copy_test_libraries)
endfunction()

# Helper function to create standalone test executables (without GTest)
function(create_standalone_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE
        contorchionist_core # Link against the core library
        ${TORCH_LIBRARIES}  # Link against Torch libraries
    )
    # Set platform-specific RPATH
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_TESTS_DIR}" # Output to build/dist/tests
            INSTALL_RPATH "@loader_path" # RPATH to find libs in the executable's own directory (macOS)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    else()
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_TESTS_DIR}" # Output to build/dist/tests
            INSTALL_RPATH "$ORIGIN" # RPATH to find libs in the executable's own directory (Linux)
            BUILD_WITH_INSTALL_RPATH TRUE # Ensure RPATH is used for build tree
        )
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name} WORKING_DIRECTORY "${PROJECT_TESTS_DIR}")

    # Make this test depend on the shared library copy target
    add_dependencies(${target_name} copy_test_libraries)
endfunction()

# Create test executables - automatically include all .cpp files not starting with _
file(GLOB TEST_SOURCES "*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_FILENAME ${TEST_SOURCE} NAME_WE)
    # Skip files that start with underscore
    if(NOT TEST_FILENAME MATCHES "^_")
        create_gtest_executable(${TEST_FILENAME} ${TEST_SOURCE})
    endif()
endforeach()

# Custom target to run all tests (optional, CTest is preferred)
# ... (your existing add_custom_target for run_all_tests can remain if you use it)
# Consider using 'ctest' command from the build directory instead.
