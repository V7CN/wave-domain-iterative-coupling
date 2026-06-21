#pragma once

#include "benchmark_config.hpp"

#include "iterative_coupling/core/types.hpp"
#include "iterative_coupling/coupling/prepared_wave_interconnection.hpp"
#include "iterative_coupling/solver/frozen_port_map.hpp"

namespace two_oscillator_benchmark {

struct RunOutput;

struct AuditRow {
  double fne_min_delta_A = 0.0;
  double fne_min_delta_B = 0.0;
  int fne_viol_A = 0;
  int fne_viol_B = 0;
  double rA = 0.0;
  double rB = 0.0;
  double augmented_residual = 0.0;
  double inner_residual = 0.0;
  int inner_iterations = 0;
};

struct ErrorSummary {
  int K = 0;
  double max_error = 0.0;
  double rms_error = 0.0;
  double final_error = 0.0;
  double max_passivity_pos = 0.0;
  double max_augmented_pos = 0.0;
  double min_fne_margin = 0.0;
};

struct FrozenDiagnostics {
  AuditRow audit;
  iterative_coupling::Wave u_dagger;
};

FrozenDiagnostics evaluate_frozen_diagnostics(
    const iterative_coupling::StackedFrozenPortMap& frozen_map,
    const iterative_coupling::PreparedWaveInterconnection& interconnection,
    const iterative_coupling::Wave& b_previous,
    int step_index,
    const DiagnosticConfig& config);

AuditRow finalize_step_diagnostics(const FrozenDiagnostics& frozen,
                                   const iterative_coupling::StepLog& step);

ErrorSummary summarize_error(int K,
                             const RunOutput& output,
                             const RunOutput& reference,
                             bool diagnostics_enabled);

}  // namespace two_oscillator_benchmark
