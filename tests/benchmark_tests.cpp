#include "benchmark_config.hpp"
#include "benchmark_io.hpp"
#include "benchmark_runner.hpp"
#include "two_oscillator_model.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

using namespace two_oscillator_benchmark;
using namespace iterative_coupling;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_formal_defaults() {
  const BenchmarkConfig config;
  check(config.physics.m1 == 8.0 && config.physics.m2 == 4.0, "mass defaults changed");
  check(config.physics.k1 == 100.0 && config.physics.k2 == 50.0, "stiffness defaults changed");
  check(config.physics.k12 == 120.0 && config.physics.c12 == 0.05 &&
            config.physics.knl == 8000.0,
        "coupling defaults changed");
  check(config.initial.q1 == 0.4 && config.initial.delta == -0.4, "initial state changed");
  check(config.dt == 0.01 && config.steps == 1000 && config.gamma == 0.4,
        "time/scattering defaults changed");
  check(config.reference_iterations == 200 && config.diagnostics.reference_iterations == 80,
        "reference iteration defaults changed");
  check(config.iteration_budgets == std::vector<int>({0, 3, 8, 20, 35, 50}),
        "finite iteration budgets changed");
  check(config.model_solver.tolerance == 1e-12 && config.model_solver.max_iterations == 25,
        "Newton defaults changed");
  check(config.execution.worker_count == 0, "parallel execution should default to automatic");
  check(config.diagnostics.fne_pairs == 32 && config.diagnostics.fne_scale == 1.0 &&
            config.diagnostics.fne_tolerance == 1e-12 &&
            config.diagnostics.fne_seed_A == 1000 && config.diagnostics.fne_seed_B == 2000,
        "FNE defaults changed");
}

void test_no_diagnostics_output_contract() {
  BenchmarkConfig config;
  config.steps = 2;
  config.reference_iterations = 2;
  config.iteration_budgets = {0};
  config.diagnostics.enabled = false;
  const BenchmarkResult result = run_benchmark(config);

  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "iterative_coupling_benchmark_unit_output";
  std::filesystem::remove_all(output);
  std::filesystem::create_directories(output);
  std::ofstream(output / "diagnostics_K0.csv") << "stale\n";
  std::ofstream(output / "trajectory_K999.csv") << "stale\n";
  std::ofstream(output / "keep_me.txt") << "unowned\n";
  write_benchmark_results(output, result);
  check(std::filesystem::is_regular_file(output / "reference_bigK.csv"),
        "reference output missing");
  check(std::filesystem::is_regular_file(output / "trajectory_K0.csv"),
        "trajectory output missing");
  check(std::filesystem::is_regular_file(output / "metadata.json"), "metadata output missing");
  check(!std::filesystem::exists(output / "diagnostics_K0.csv"),
        "--no-diagnostics contract emitted diagnostics CSV");
  check(!std::filesystem::exists(output / "trajectory_K999.csv"),
        "stale K-indexed trajectory survived output replacement");
  check(std::filesystem::is_regular_file(output / "keep_me.txt"),
        "output cleanup removed a file it does not own");

  std::ifstream stream(output / "summary.json");
  const std::string summary((std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
  check(summary.find("max_error") != std::string::npos, "position error summary missing");
  check(summary.find("max_passivity_pos") == std::string::npos &&
            summary.find("max_augmented_pos") == std::string::npos &&
            summary.find("min_fne_margin") == std::string::npos,
        "disabled diagnostics leaked into summary schema");

  std::ifstream metadata_stream(output / "metadata.json");
  const std::string metadata((std::istreambuf_iterator<char>(metadata_stream)),
                             std::istreambuf_iterator<char>());
  check(metadata.find("\"schema_version\": 3") != std::string::npos,
        "metadata schema was not upgraded to v3");
  check(metadata.find("\"type\": \"linear-dirac\"") != std::string::npos &&
            metadata.find("\"factory\": \"equal-effort-opposite-flow\"") !=
                std::string::npos &&
            metadata.find("\"wave_orthogonality_residual\"") != std::string::npos,
        "metadata omitted linear Dirac interconnection provenance");
  check(metadata.find(std::string("coupling_") + "sign") == std::string::npos,
        "obsolete coupling field survived in metadata");
  std::filesystem::remove_all(output);
}

void test_two_oscillator_model_frozen_step_baseline() {
  const BenchmarkConfig config;
  const auto model = two_oscillator_example::make_two_oscillator_model(
      config.physics, config.initial, config.model_solver);
  const StepContext context{config.dt, config.gamma, 0};
  struct ExpectedStep {
    double incident;
    std::array<double, 3> xA;
    double bA;
    std::array<double, 2> xB;
    double bB;
  };
  const std::array<ExpectedStep, 3> expected{{
      {-0.4,
       {0.39646568590396247, -0.70686281920750449, 0.024415845712779355},
       -4.1644788606770478,
       {4.4685052944471185e-05, 0.0089370105888957278},
       -0.39960032473615187},
      {0.0,
       {0.39647935967799208, -0.70412806440159137, 0.058484053456003007},
       -4.0693164761717355,
       {0.0, 0.0},
       0.0},
      {0.35,
       {0.39649132422424765, -0.70173515515047757, 0.088293735233617138},
       -3.9860493896962521,
       {-3.909942132638454e-05, -0.0078198842652836831},
       0.34965028414413296},
  }};
  for (const ExpectedStep& baseline : expected) {
    Wave a(1);
    a << baseline.incident;
    const StepResult result_A = model.phi_A.step(model.x_A0, a, context);
    const StepResult result_B = model.phi_B.step(model.x_B0, a, context);
    const State expected_A = Eigen::Map<const State>(baseline.xA.data(), 3);
    const State expected_B = Eigen::Map<const State>(baseline.xB.data(), 2);
    check((result_A.x_next - expected_A).norm() < 1e-12 &&
              std::abs(result_A.b_next(0) - baseline.bA) < 1e-12,
          "subsystem A changed from the generalized-pH frozen-step baseline");
    check((result_B.x_next - expected_B).norm() < 1e-12 &&
              std::abs(result_B.b_next(0) - baseline.bB) < 1e-12,
          "subsystem B changed from the generalized-pH frozen-step baseline");
    check(result_A.solver_stats.iterations > 1,
          "subsystem A unexpectedly bypassed analytic Newton");
    check(result_B.solver_stats.iterations == 1,
          "subsystem B did not use the affine direct path");
  }
}

void test_removed_model_backend_cli_is_rejected() {
  char program[] = "two_oscillators_benchmark";
  char output[] = "unused";
  char flag[] = "--model-backend";
  char value[] = "generalized-ph";
  char* argv[] = {program, output, flag, value};
  bool rejected = false;
  try {
    (void)parse_command_line(4, argv);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "removed --model-backend CLI option was still accepted");
}

}  // namespace

int main() {
  test_formal_defaults();
  test_no_diagnostics_output_contract();
  test_two_oscillator_model_frozen_step_baseline();
  test_removed_model_backend_cli_is_rejected();
  std::cout << "All two-oscillator benchmark tests passed\n";
  return 0;
}
