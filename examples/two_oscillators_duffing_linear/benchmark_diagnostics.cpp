#include "benchmark_diagnostics.hpp"

#include "benchmark_runner.hpp"

#include "iterative_coupling/solver/drs_interface_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace two_oscillator_benchmark {
namespace {

std::pair<double, int> fne_sampling_check(const iterative_coupling::FrozenPortMap& map,
                                          int n_pairs,
                                          int seed,
                                          double scale,
                                          double tolerance) {
  // Offline empirical certificate for Condition 1. This is not
  // an online step of Algorithm 1 and does not constitute an analytical proof.
  if (map.port_dimension() != 1) {
    throw std::invalid_argument("Scalar FNE sampler requires a one-dimensional port map");
  }
  std::mt19937 rng(static_cast<unsigned int>(seed));
  std::normal_distribution<double> normal(0.0, scale);
  double min_delta = std::numeric_limits<double>::infinity();
  int violations = 0;
  for (int i = 0; i < n_pairs; ++i) {
    iterative_coupling::Wave a(1);
    iterative_coupling::Wave ap(1);
    a(0) = normal(rng);
    ap(0) = normal(rng);
    const iterative_coupling::Wave b = map(a);
    const iterative_coupling::Wave bp = map(ap);
    const double db = b(0) - bp(0);
    const double da = a(0) - ap(0);
    const double delta = db * da - db * db;
    min_delta = std::min(min_delta, delta);
    if (delta < -tolerance) {
      ++violations;
    }
  }
  return {min_delta, violations};
}

double augmented_storage_residual(double H_n,
                                  double H_np1,
                                  const iterative_coupling::Wave& u0,
                                  const iterative_coupling::Wave& uK,
                                  const iterative_coupling::Wave& u_dagger,
                                  const iterative_coupling::Wave& a_hat,
                                  const iterative_coupling::Wave& b_np1) {
  // Offline evaluation of Eq. (31). The long-DRS
  // approximation u_dagger is diagnostic-only and is not part of Algorithm 1.
  const double err0 = 0.5 * (u0 - u_dagger).squaredNorm();
  const double errK = 0.5 * (uK - u_dagger).squaredNorm();
  const double supply = 0.5 * (a_hat.squaredNorm() - b_np1.squaredNorm());
  return (H_np1 + errK) - (H_n + err0) - supply;
}

}  // namespace

FrozenDiagnostics evaluate_frozen_diagnostics(
    const iterative_coupling::StackedFrozenPortMap& frozen_map,
    const iterative_coupling::PreparedWaveInterconnection& interconnection,
    const iterative_coupling::Wave& b_previous,
    int step_index,
    const DiagnosticConfig& config) {
  if (!config.enabled || frozen_map.size() != 2) {
    throw std::invalid_argument("Enabled two-oscillator diagnostics require two maps");
  }

  FrozenDiagnostics out;
  const auto [minA, violA] =
      fne_sampling_check(frozen_map.map(0), config.fne_pairs, config.fne_seed_A + step_index,
                         config.fne_scale, config.fne_tolerance);
  const auto [minB, violB] =
      fne_sampling_check(frozen_map.map(1), config.fne_pairs, config.fne_seed_B + step_index,
                         config.fne_scale, config.fne_tolerance);
  out.audit.fne_min_delta_A = minA;
  out.audit.fne_min_delta_B = minB;
  out.audit.fne_viol_A = violA;
  out.audit.fne_viol_B = violB;

  iterative_coupling::DrsOptions options;
  options.K_n = config.reference_iterations;
  const iterative_coupling::DrsResult reference = iterative_coupling::DrsInterfaceSolver{}.solve(
      frozen_map, interconnection, b_previous, options);
  out.u_dagger = reference.uK;
  return out;
}

AuditRow finalize_step_diagnostics(const FrozenDiagnostics& frozen,
                                   const iterative_coupling::StepLog& step) {
  if (step.energy.size() != 2) {
    throw std::invalid_argument("Two-oscillator diagnostics require two subsystem energy logs");
  }
  AuditRow row = frozen.audit;
  row.rA = step.energy[0].passivity_residual;
  row.rB = step.energy[1].passivity_residual;
  const double H_n = step.energy[0].h_before + step.energy[1].h_before;
  const double H_np1 = step.energy[0].h_after + step.energy[1].h_after;
  row.augmented_residual = augmented_storage_residual(
      H_n, H_np1, step.u0, step.uK, frozen.u_dagger, step.a_used, step.b_next);
  row.inner_iterations = step.inner_iterations;
  row.inner_residual = step.inner_residual;
  return row;
}

ErrorSummary summarize_error(int K,
                             const RunOutput& output,
                             const RunOutput& reference,
                             bool diagnostics_enabled) {
  if (output.t.size() != reference.t.size()) {
    throw std::invalid_argument("Cannot compare runs with different time-grid lengths");
  }
  ErrorSummary summary;
  summary.K = K;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < output.t.size(); ++i) {
    const double e1 = output.xA[i](0) - reference.xA[i](0);
    const double e2 = output.xB[i](0) - reference.xB[i](0);
    const double error = std::sqrt(e1 * e1 + e2 * e2);
    summary.max_error = std::max(summary.max_error, error);
    sum_sq += error * error;
    if (i + 1 == output.t.size()) {
      summary.final_error = error;
    }
  }
  summary.rms_error =
      output.t.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(output.t.size()));

  if (diagnostics_enabled) {
    summary.min_fne_margin = std::numeric_limits<double>::infinity();
    for (const AuditRow& row : output.audit) {
      summary.max_passivity_pos =
          std::max(summary.max_passivity_pos, std::max({0.0, row.rA, row.rB}));
      summary.max_augmented_pos =
          std::max(summary.max_augmented_pos, std::max(0.0, row.augmented_residual));
      summary.min_fne_margin =
          std::min(summary.min_fne_margin,
                   std::min(row.fne_min_delta_A, row.fne_min_delta_B));
    }
  }
  return summary;
}

}  // namespace two_oscillator_benchmark
