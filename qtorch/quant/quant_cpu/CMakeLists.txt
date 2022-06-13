cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(qtorch_torchscript)

find_package(Torch REQUIRED)

# Define our library target
add_library(qtorch_ops SHARED
    quant_cpu.cpp
    bit_helper.cpp
    sim_helper.cpp
    quant_cpu.h
)
# Enable C++14
target_compile_features(qtorch_ops PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(qtorch_ops "${TORCH_LIBRARIES}")


