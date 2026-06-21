#include "benchmark_config.hpp"

#include <stdexcept>
#include <string>

namespace two_oscillator_benchmark {

void BenchmarkConfig::validate() const {
  model_solver.validate();
  if (dt <= 0.0 || steps < 0 || gamma <= 0.0 || reference_iterations < 0) {
    throw std::invalid_argument("Invalid benchmark time, DRS, or reference configuration");
  }
  if (iteration_budgets.empty()) {
    throw std::invalid_argument("Benchmark requires at least one finite iteration budget");
  }
  for (int K : iteration_budgets) {
    if (K < 0) {
      throw std::invalid_argument("Benchmark iteration budgets must be non-negative");
    }
  }
  if (diagnostics.enabled &&
      (diagnostics.reference_iterations < 0 || diagnostics.fne_pairs <= 0 ||
       diagnostics.fne_scale <= 0.0 || diagnostics.fne_tolerance < 0.0)) {
    throw std::invalid_argument("Invalid benchmark diagnostic configuration");
  }
}

CommandLineOptions parse_command_line(int argc, char** argv) {
  CommandLineOptions options;
  if (argc >= 2) {
    options.output_directory = argv[1];
  }

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) {
      if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      return std::string(argv[++i]);
    };
    if (arg == "--steps") {
      options.benchmark.steps = std::stoi(require_value(arg));
    } else if (arg == "--refK") {
      options.benchmark.reference_iterations = std::stoi(require_value(arg));
    } else if (arg == "--diagK") {
      options.benchmark.diagnostics.reference_iterations = std::stoi(require_value(arg));
    } else if (arg == "--fne-pairs") {
      options.benchmark.diagnostics.fne_pairs = std::stoi(require_value(arg));
    } else if (arg == "--workers") {
      const int workers = std::stoi(require_value(arg));
      if (workers < 0) {
        throw std::invalid_argument("--workers must be non-negative");
      }
      options.benchmark.execution.worker_count = static_cast<std::size_t>(workers);
    } else if (arg == "--no-diagnostics") {
      options.benchmark.diagnostics.enabled = false;
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }
  options.benchmark.validate();
  return options;
}

}  // namespace two_oscillator_benchmark
