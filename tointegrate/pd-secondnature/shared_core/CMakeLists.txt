cmake_minimum_required(VERSION 3.15)
project(SharedCore)

add_library(SharedCore STATIC
    src/rms_analyzer.cpp
    src/audio_utils.cpp
    src/windowing.cpp
)

target_include_directories(SharedCore PUBLIC
    include
)
