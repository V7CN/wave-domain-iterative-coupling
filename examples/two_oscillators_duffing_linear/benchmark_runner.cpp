#include "benchmark_runner.hpp"
#include "two_oscillator_model.hpp"

#include "iterative_coupling/coupling/linear_dirac_interconnection.hpp"
#include "iterative_coupling/solver/partitioned_simulator.hpp"

#include <iostream>
#include <utility>

namespace two_oscillator_benchmark {

RunOutput run_case(const BenchmarkConfig& config, int K, bool diagnostics_enabled) {
  config.validate();
  const two_oscillator_example::TwoOscillatorModel model =
      two_oscillator_example::make_two_oscillator_model(
          config.physics, config.initial, config.model_solver);
  const iterative_coupling::LinearDiracInterconnection interconnection =
      iterative_coupling::LinearDiracInterconnection::equal_effort_opposite_flow(1);

  iterative_coupling::DrsOptions drs_options;
  drs_options.K_n = K;
  iterative_coupling::PartitionedSimulator simulator(
      {{model.phi_A, model.x_A0}, {model.phi_B, model.x_B0}}, model.b0, config.dt, config.gamma,
      interconnection, drs_options,
      config.execution);

  RunOutput output;
  output.worker_count = simulator.worker_count();
  output.interconnection_diagnostics = simulator.prepared_interconnection().diagnostics();
  const auto sample_count = static_cast<std::size_t>(config.steps) + 1U;
  output.t.reserve(sample_count);
  output.xA.reserve(sample_count);
  output.xB.reserve(sample_count);
  output.b_grid.reserve(sample_count);
  if (diagnostics_enabled) {
    output.audit.reserve(static_cast<std::size_t>(config.steps));
  }
  output.t.push_back(0.0);
  output.xA.push_back(model.x_A0);
  output.xB.push_back(model.x_B0);
  output.b_grid.push_back(model.b0);

  for (int n = 0; n < config.steps; ++n) {
    FrozenDiagnostics frozen;
    iterative_coupling::PartitionedSimulator::FrozenStepObserver observer;
    if (diagnostics_enabled) {
      observer = [&](int observed_step,
                     const iterative_coupling::StackedFrozenPortMap& frozen_map,
                     const iterative_coupling::PreparedWaveInterconnection&
                         observed_interconnection,
                     const iterative_coupling::Wave& b_previous) {
        frozen = evaluate_frozen_diagnostics(frozen_map, observed_interconnection, b_previous,
                                             observed_step, config.diagnostics);
      };
    }

    const iterative_coupling::StepLog step = simulator.step(n, observer);
    output.t.push_back(static_cast<double>(n + 1) * config.dt);
    output.xA.push_back(step.x_after.at(0));
    output.xB.push_back(step.x_after.at(1));
    output.b_grid.push_back(step.b_next);
    if (diagnostics_enabled) {
      output.audit.push_back(finalize_step_diagnostics(frozen, step));
    }
  }
  return output;
}

BenchmarkResult run_benchmark(const BenchmarkConfig& config) {
  config.validate();
  BenchmarkResult result;
  result.config = config;

  std::cout << "Model formulation: " << two_oscillator_example::kModelProvenance << "\n";
  std::cout << "Running hard-coupling bigK reference"
            << " (steps=" << config.steps << ", refK=" << config.reference_iterations << ")..."
            << std::endl;
  result.reference = run_case(config, config.reference_iterations, false);

  for (int K : config.iteration_budgets) {
    std::cout << "Running K_n=" << K;
    if (config.diagnostics.enabled) {
      std::cout << " (diagK=" << config.diagnostics.reference_iterations
                << ", diagnostics=on)";
    } else {
      std::cout << " (diagnostics=off)";
    }
    std::cout << "..." << std::endl;

    CaseOutput current;
    current.K = K;
    current.run = run_case(config, K, config.diagnostics.enabled);
    result.summaries.push_back(
        summarize_error(K, current.run, result.reference, config.diagnostics.enabled));
    result.cases.push_back(std::move(current));
  }
  return result;
}

}  // namespace two_oscillator_benchmark
