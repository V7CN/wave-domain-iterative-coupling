#pragma once

#include "two_oscillator_model.hpp"

#include "iterative_coupling/core/parallel_executor.hpp"

#include <filesystem>
#include <vector>

namespace two_oscillator_benchmark {

struct DiagnosticConfig {
  bool enabled = true;
  int reference_iterations = 80;
  int fne_pairs = 32;
  double fne_scale = 1.0;
  double fne_tolerance = 1e-12;
  int fne_seed_A = 1000;
  int fne_seed_B = 2000;
};

struct BenchmarkConfig {
  two_oscillator_example::PhysicalParameters physics;
  two_oscillator_example::SolverOptions model_solver;
  two_oscillator_example::InitialConditions initial;
  double dt = 0.01;
  int steps = 1000;
  double gamma = 0.4;
  iterative_coupling::ExecutionOptions execution;
  int reference_iterations = 200;
  std::vector<int> iteration_budgets{0, 3, 8, 20, 35, 50};
  DiagnosticConfig diagnostics;

  void validate() const;
};

struct CommandLineOptions {
  std::filesystem::path output_directory = "examples/two_oscillators_duffing_linear/out_cpp";
  BenchmarkConfig benchmark;
};

CommandLineOptions parse_command_line(int argc, char** argv);

}  // namespace two_oscillator_benchmark
