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

add_executable(sts_action_transition_test EXCLUDE_FROM_ALL tests/action_transition_test.cpp)
target_link_libraries(sts_action_transition_test PRIVATE sts_lightspeed)

option(STS_LIGHTSPEED_BUILD_TESTS "Build and register simulator regression tests" OFF)
if(STS_LIGHTSPEED_BUILD_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS sts_liquid_memories_test sts_action_transition_test)
        set_target_properties(${test_name} PROPERTIES EXCLUDE_FROM_ALL FALSE)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
    endforeach()
endif()
