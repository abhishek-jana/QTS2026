cmake_minimum_required(VERSION 3.10)
project(UQTS_Execution_Muscle)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch (Assuming installed or path provided)
# find_package(Torch REQUIRED)

add_executable(muscle main.cpp)
# target_link_libraries(muscle "${TORCH_LIBRARIES}")
