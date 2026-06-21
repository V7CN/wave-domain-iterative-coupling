cmake_minimum_required(VERSION 3.16)
project(iterative_coupling_new LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATH_SUFFIXES eigen3)
  if(EIGEN3_INCLUDE_DIR)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN3_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Eigen3 not found. Install Eigen or set Eigen3_DIR/CMAKE_PREFIX_PATH.")
  endif()
endif()

add_library(iterative_coupling INTERFACE)
target_include_directories(iterative_coupling INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(iterative_coupling INTERFACE Eigen3::Eigen Threads::Threads)

add_library(two_oscillators_benchmark_support STATIC
  examples/two_oscillators_duffing_linear/two_oscillator_model.cpp
  examples/two_oscillators_duffing_linear/benchmark_config.cpp
  examples/two_oscillators_duffing_linear/benchmark_diagnostics.cpp
  examples/two_oscillators_duffing_linear/benchmark_io.cpp
  examples/two_oscillators_duffing_linear/benchmark_runner.cpp
)
target_include_directories(two_oscillators_benchmark_support PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/two_oscillators_duffing_linear
)
target_link_libraries(two_oscillators_benchmark_support PUBLIC iterative_coupling)

add_executable(two_oscillators_benchmark
  examples/two_oscillators_duffing_linear/main.cpp
)
target_link_libraries(two_oscillators_benchmark PRIVATE two_oscillators_benchmark_support)

enable_testing()
add_executable(iterative_coupling_tests
  tests/core_tests.cpp
)
target_link_libraries(iterative_coupling_tests PRIVATE iterative_coupling)
add_test(NAME iterative_coupling_tests COMMAND iterative_coupling_tests)

add_executable(two_oscillators_benchmark_tests
  tests/benchmark_tests.cpp
)
target_link_libraries(two_oscillators_benchmark_tests PRIVATE two_oscillators_benchmark_support)
add_test(NAME two_oscillators_benchmark_tests COMMAND two_oscillators_benchmark_tests)

add_executable(generic_ph_mass_spring
  examples/generic_ph_mass_spring/main.cpp
)
target_link_libraries(generic_ph_mass_spring PRIVATE iterative_coupling)

add_executable(iterative_coupling_ph_tests
  tests/ph_tests.cpp
)
target_link_libraries(iterative_coupling_ph_tests PRIVATE iterative_coupling)
add_test(NAME iterative_coupling_ph_tests COMMAND iterative_coupling_ph_tests)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(two_oscillators_benchmark_support PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(two_oscillators_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(iterative_coupling_tests PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(two_oscillators_benchmark_tests PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(generic_ph_mass_spring PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(iterative_coupling_ph_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
