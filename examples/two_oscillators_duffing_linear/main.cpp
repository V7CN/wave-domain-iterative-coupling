#include "benchmark_config.hpp"
#include "benchmark_io.hpp"
#include "benchmark_runner.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const two_oscillator_benchmark::CommandLineOptions options =
        two_oscillator_benchmark::parse_command_line(argc, argv);
    const two_oscillator_benchmark::BenchmarkResult result =
        two_oscillator_benchmark::run_benchmark(options.benchmark);
    two_oscillator_benchmark::write_benchmark_results(options.output_directory, result);
    two_oscillator_benchmark::print_benchmark_summary(options.output_directory, result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
