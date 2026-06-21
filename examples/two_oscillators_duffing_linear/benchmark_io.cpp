#include "benchmark_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace two_oscillator_benchmark {
namespace {

std::ofstream open_output(const std::filesystem::path& path) {
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("Could not open output file: " + path.string());
  }
  stream << std::setprecision(17);
  return stream;
}

bool is_indexed_csv(const std::string& name, const std::string& prefix) {
  constexpr const char* suffix = ".csv";
  if (name.size() <= prefix.size() + 4 || name.compare(0, prefix.size(), prefix) != 0 ||
      name.compare(name.size() - 4, 4, suffix) != 0) {
    return false;
  }
  return std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                     name.end() - 4,
                     [](unsigned char character) { return std::isdigit(character) != 0; });
}

bool is_owned_output(const std::filesystem::path& path) {
  const std::string name = path.filename().string();
  return name == "reference_bigK.csv" || name == "summary.json" || name == "metadata.json" ||
         is_indexed_csv(name, "trajectory_K") || is_indexed_csv(name, "diagnostics_K");
}

void remove_owned_outputs(const std::filesystem::path& output_directory) {
  if (!std::filesystem::exists(output_directory)) {
    return;
  }
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(output_directory)) {
    if (entry.is_regular_file() && is_owned_output(entry.path())) {
      std::filesystem::remove(entry.path());
    }
  }
}

void write_trajectory_csv(const std::filesystem::path& path, const RunOutput& output) {
  std::ofstream stream = open_output(path);
  stream << "t,q1,v1,delta,q2,v2,bA,bB\n";
  for (std::size_t i = 0; i < output.t.size(); ++i) {
    stream << output.t[i] << "," << output.xA[i](0) << "," << output.xA[i](1) << ","
           << output.xA[i](2) << "," << output.xB[i](0) << "," << output.xB[i](1) << ","
           << output.b_grid[i](0) << "," << output.b_grid[i](1) << "\n";
  }
}

void write_diagnostics_csv(const std::filesystem::path& path, const RunOutput& output) {
  std::ofstream stream = open_output(path);
  stream << "n,fne_min_delta_A,fne_min_delta_B,fne_viol_A,fne_viol_B,rA,rB,"
            "augmented_residual,inner_residual,inner_iterations\n";
  for (std::size_t n = 0; n < output.audit.size(); ++n) {
    const AuditRow& row = output.audit[n];
    stream << n << "," << row.fne_min_delta_A << "," << row.fne_min_delta_B << ","
           << row.fne_viol_A << "," << row.fne_viol_B << "," << row.rA << "," << row.rB
           << "," << row.augmented_residual << "," << row.inner_residual << ","
           << row.inner_iterations << "\n";
  }
}

void write_summary_json(const std::filesystem::path& path, const BenchmarkResult& result) {
  std::ofstream stream = open_output(path);
  stream << "{\n  \"summaries\": [\n";
  for (std::size_t i = 0; i < result.summaries.size(); ++i) {
    const ErrorSummary& summary = result.summaries[i];
    stream << "    {\"K_n\": " << summary.K << ", \"max_error\": " << summary.max_error
           << ", \"rms_error\": " << summary.rms_error
           << ", \"final_error\": " << summary.final_error;
    if (result.config.diagnostics.enabled) {
      stream << ", \"max_passivity_pos\": " << summary.max_passivity_pos
             << ", \"max_augmented_pos\": " << summary.max_augmented_pos
             << ", \"min_fne_margin\": " << summary.min_fne_margin;
    }
    stream << "}";
    if (i + 1 != result.summaries.size()) {
      stream << ",";
    }
    stream << "\n";
  }
  stream << "  ]\n}\n";
}

void write_metadata_json(const std::filesystem::path& path, const BenchmarkResult& result) {
  std::ofstream stream = open_output(path);
  const BenchmarkConfig& c = result.config;
  const iterative_coupling::InterconnectionDiagnostics& interconnection =
      result.reference.interconnection_diagnostics;
  stream << "{\n"
         << "  \"schema_version\": 3,\n"
         << "  \"model_backend\": \"" << two_oscillator_example::kModelProvenance
         << "\",\n"
         << "  \"dt\": " << c.dt << ", \"steps\": " << c.steps
         << ", \"gamma\": " << c.gamma << ",\n"
         << "  \"interconnection\": {\"type\": \"linear-dirac\", "
            "\"factory\": \"equal-effort-opposite-flow\", "
            "\"total_port_dimension\": "
         << interconnection.constraint_rank << ", \"constraint_rank\": "
         << interconnection.constraint_rank << ", \"dirac_residual\": "
         << interconnection.dirac_residual << ", \"wave_orthogonality_residual\": "
         << interconnection.wave_orthogonality_residual
         << ", \"resolvent_condition_number\": "
         << interconnection.resolvent_condition_number << "},\n"
         << "  \"reference_iterations\": " << c.reference_iterations << ",\n"
         << "  \"iteration_budgets\": [";
  for (std::size_t i = 0; i < c.iteration_budgets.size(); ++i) {
    if (i != 0) stream << ", ";
    stream << c.iteration_budgets[i];
  }
  stream << "],\n"
         << "  \"diagnostics\": {\"enabled\": "
         << (c.diagnostics.enabled ? "true" : "false")
         << ", \"reference_iterations\": " << c.diagnostics.reference_iterations
         << ", \"fne_pairs\": " << c.diagnostics.fne_pairs
         << ", \"fne_scale\": " << c.diagnostics.fne_scale
         << ", \"fne_tolerance\": " << c.diagnostics.fne_tolerance
         << ", \"fne_seed_A\": " << c.diagnostics.fne_seed_A
         << ", \"fne_seed_B\": " << c.diagnostics.fne_seed_B << "},\n"
         << "  \"execution\": {\"requested_workers\": " << c.execution.worker_count
         << ", \"effective_workers\": " << result.reference.worker_count << "},\n"
         << "  \"physics\": {\"m1\": " << c.physics.m1 << ", \"m2\": " << c.physics.m2
         << ", \"k1\": " << c.physics.k1 << ", \"k2\": " << c.physics.k2
         << ", \"c1\": " << c.physics.c1 << ", \"c2\": " << c.physics.c2
         << ", \"k12\": " << c.physics.k12 << ", \"c12\": " << c.physics.c12
         << ", \"knl\": " << c.physics.knl << "},\n"
         << "  \"initial\": {\"q1\": " << c.initial.q1 << ", \"v1\": " << c.initial.v1
         << ", \"delta\": " << c.initial.delta << ", \"q2\": " << c.initial.q2
         << ", \"v2\": " << c.initial.v2 << ", \"bA\": " << c.initial.bA
         << ", \"bB\": " << c.initial.bB << "},\n"
         << "  \"model_solver\": {\"tolerance\": " << c.model_solver.tolerance
         << ", \"max_iterations\": " << c.model_solver.max_iterations
         << ", \"max_line_search_iterations\": "
         << c.model_solver.max_line_search_iterations << ", \"armijo_factor\": "
         << c.model_solver.armijo_factor << "}\n"
         << "}\n";
}

}  // namespace

void write_benchmark_results(const std::filesystem::path& output_directory,
                             const BenchmarkResult& result) {
  std::filesystem::create_directories(output_directory);
  remove_owned_outputs(output_directory);
  write_trajectory_csv(output_directory / "reference_bigK.csv", result.reference);
  for (const CaseOutput& current : result.cases) {
    write_trajectory_csv(output_directory / ("trajectory_K" + std::to_string(current.K) + ".csv"),
                         current.run);
    if (result.config.diagnostics.enabled) {
      write_diagnostics_csv(
          output_directory / ("diagnostics_K" + std::to_string(current.K) + ".csv"), current.run);
    }
  }
  write_summary_json(output_directory / "summary.json", result);
  write_metadata_json(output_directory / "metadata.json", result);
}

void print_benchmark_summary(const std::filesystem::path& output_directory,
                             const BenchmarkResult& result) {
  std::cout << "\nPosition error vs hard-coupling discrete-port reference\n";
  for (const ErrorSummary& summary : result.summaries) {
    std::cout << "K_n=" << std::setw(2) << summary.K << " | max=" << std::scientific
              << summary.max_error << " | rms=" << summary.rms_error
              << " | final=" << summary.final_error;
    if (result.config.diagnostics.enabled) {
      std::cout << " | max_passivity+=" << summary.max_passivity_pos
                << " | max_augmented+=" << summary.max_augmented_pos;
    }
    std::cout << "\n";
  }
  std::cout << "Wrote outputs to " << output_directory << "\n";
}

}  // namespace two_oscillator_benchmark
