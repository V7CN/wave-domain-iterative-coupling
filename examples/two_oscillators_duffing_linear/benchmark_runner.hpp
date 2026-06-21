#pragma once

#include "benchmark_diagnostics.hpp"

#include "iterative_coupling/core/types.hpp"

#include <vector>

namespace two_oscillator_benchmark {

struct RunOutput {
  std::size_t worker_count = 1;
  iterative_coupling::InterconnectionDiagnostics interconnection_diagnostics;
  std::vector<double> t;
  std::vector<iterative_coupling::State> xA;
  std::vector<iterative_coupling::State> xB;
  std::vector<iterative_coupling::Wave> b_grid;
  std::vector<AuditRow> audit;
};

struct CaseOutput {
  int K = 0;
  RunOutput run;
};

struct BenchmarkResult {
  BenchmarkConfig config;
  RunOutput reference;
  std::vector<CaseOutput> cases;
  std::vector<ErrorSummary> summaries;
};

RunOutput run_case(const BenchmarkConfig& config, int K, bool diagnostics_enabled);
BenchmarkResult run_benchmark(const BenchmarkConfig& config);

}  // namespace two_oscillator_benchmark
