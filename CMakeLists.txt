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

add_executable(search-fusion-smoke tests/search_fusion_smoke.cpp)
target_link_libraries(search-fusion-smoke PRIVATE sts_lightspeed)
enable_testing()
add_test(NAME search-fusion-smoke COMMAND search-fusion-smoke)
add_executable(search-fusion-benchmark tests/search_fusion_benchmark.cpp)
target_link_libraries(search-fusion-benchmark PRIVATE sts_lightspeed)
