cmake_minimum_required(VERSION 3.19)
project(sts_lightspeed LANGUAGES CXX)

file(GLOB_RECURSE STS_LIGHTSPEED_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(sts_lightspeed STATIC ${STS_LIGHTSPEED_SOURCES})
target_include_directories(sts_lightspeed SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/json/include")
target_compile_features(sts_lightspeed PUBLIC cxx_std_17)
set_target_properties(sts_lightspeed PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sts_lightspeed PRIVATE -Wno-shift-count-overflow)
endif()

# Opt-in regression executable; production/library builds do not build it by default.
add_executable(sts_liquid_memories_test EXCLUDE_FROM_ALL tests/liquid_memories_test.cpp)
target_link_libraries(sts_liquid_memories_test PRIVATE sts_lightspeed)
