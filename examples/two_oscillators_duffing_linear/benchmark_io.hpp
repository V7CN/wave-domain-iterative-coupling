#pragma once

#include "benchmark_runner.hpp"

#include <filesystem>

namespace two_oscillator_benchmark {

void write_benchmark_results(const std::filesystem::path& output_directory,
                             const BenchmarkResult& result);
void print_benchmark_summary(const std::filesystem::path& output_directory,
                             const BenchmarkResult& result);

}  // namespace two_oscillator_benchmark
