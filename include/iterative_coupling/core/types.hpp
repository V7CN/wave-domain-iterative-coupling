#pragma once

#include <Eigen/Dense>

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace iterative_coupling {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using State = Eigen::VectorXd;

/// Discrete wave coordinate used by the interface solver.
///
/// A Wave already includes the paper's sqrt(Delta t) scaling. Consequently,
/// 0.5 * (||a||^2 - ||b||^2) is an energy supplied through the port over one
/// macro-step, not an instantaneous power.
using Wave = Eigen::VectorXd;

/// Physical effort/flow pair with supplied power e^T f under the chosen port orientation.
struct PortVariables {
  Vector effort;
  Vector flow;
};

/// Numerical information returned by one evaluation of a subsystem integrator.
struct SolverStats {
  int iterations = 0;
  double residual_norm = 0.0;
};

/// Immutable parameter snapshot for one macro-step n.
///
/// The same dt and gamma must be used by every frozen-map query and by the final
/// committed advancement of that macro-step. Changing either value inside the
/// DRS inner iteration would change S^n and invalidate the fixed-operator analysis.
struct StepContext {
  double dt = 0.0;
  double gamma = 0.0;
  int step_index = 0;

  void validate() const {
    if (dt <= 0.0) {
      throw std::invalid_argument("StepContext.dt must be positive");
    }
    if (gamma <= 0.0) {
      throw std::invalid_argument("StepContext.gamma must be positive");
    }
  }
};

/// Complete output of Phi_i^{Delta t}(x_i^n, a_i).
///
/// This result is deliberately richer than a frozen-port-map response.
/// S_i^n(a_i) exposes only b_next and discards x_next; the final macro-step
/// keeps both x_next and b_next. See Eqs. (11) and (12).
struct StepResult {
  State x_next;
  Wave b_next;
  std::optional<PortVariables> port_stage;
  std::optional<double> h_next;
  SolverStats solver_stats;
};

/// Per-subsystem residual for Condition 2 (discrete passivity).
struct EnergyDiagnostics {
  double h_before = 0.0;
  double h_after = 0.0;
  double supply = 0.0;
  double passivity_residual = 0.0;
};

enum class DrsTerminationReason {
  ZeroBudget,
  IterationBudget,
  ResidualTolerance,
  UserPredicate,
};

/// Trace of one completed macro-step, using the symbols of Algorithm 1.
struct StepLog {
  int step_index = 0;
  double time = 0.0;
  double gamma_used = 0.0;
  int K_n = 0;
  int inner_iterations = 0;
  double inner_residual = 0.0;
  DrsTerminationReason termination_reason = DrsTerminationReason::ZeroBudget;

  std::vector<State> x_before;
  std::vector<State> x_after;
  Wave b_prev;
  /// Algorithm 1 auxiliary iterate u^{n,0}; retained for offline diagnostics.
  Wave u0;
  /// Algorithm 1 auxiliary iterate u^{n,K_eff}; never used as the final incident wave.
  Wave uK;
  Wave b_hat_star;
  Wave a_used;
  Wave b_next;
  std::vector<EnergyDiagnostics> energy;
  std::vector<SolverStats> subsystem_solver_stats;
};

struct SimulationLog {
  std::vector<StepLog> steps;
};

inline void require_size(const Vector& v, Eigen::Index expected, const std::string& name) {
  if (v.size() != expected) {
    throw std::invalid_argument(name + " has dimension " + std::to_string(v.size()) +
                                ", expected " + std::to_string(expected));
  }
}

}  // namespace iterative_coupling
