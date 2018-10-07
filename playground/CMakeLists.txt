cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(playground)

set(CMAKE_PREFIX_PATH ~/Documents/cpp_lib/libtorch)

find_package(Torch REQUIRED)

add_executable(playground playground.cpp)
target_link_libraries(playground "${TORCH_LIBRARIES}")
set_property(TARGET playground PROPERTY CXX_STANDARD 11)
