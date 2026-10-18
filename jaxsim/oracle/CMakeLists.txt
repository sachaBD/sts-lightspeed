# Standalone build of the parity oracle: cmake -S jaxsim/oracle -B jaxsim/oracle/build && cmake --build ...
# (tools/build_oracle.sh does this). Not part of the main sts_lightspeed build.
cmake_minimum_required(VERSION 3.19)
project(jaxsim_oracle LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_CURRENT_BINARY_DIR}/sts_lightspeed")
add_executable(jaxsim_oracle oracle.cpp)
target_link_libraries(jaxsim_oracle PRIVATE sts_lightspeed)
