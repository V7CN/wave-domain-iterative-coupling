#pragma once

#include "iterative_coupling/coupling/prepared_wave_interconnection.hpp"
#include "iterative_coupling/solver/frozen_port_map.hpp"

namespace iterative_coupling {

/// Read-only values from one completed DRS inner iteration. A user predicate
/// may inspect these values but cannot alter the accepted algorithm state.
struct DrsIterationView {
  int k = 0;
  const Wave& u_current;
  const Wave& u_next;
  const Wave& b_hat;
  const Wave& w;
  double residual = 0.0;
};

/// Finite inner-iteration policy for macro-step n in Algorithm 1.
struct DrsOptions {
  using StopPredicate = std::function<bool(const DrsIterationView&)>;
  /// Maximum inner-iteration budget K_n. K_n=0 is the explicit-wave case.
  int K_n = 0;
  /// Optional threshold for ||u^{n,k+1}-u^{n,k}||; zero disables early termination.
  double eps = 0.0;
  /// Retain the auxiliary and shadow sequences for diagnostics or research analysis.
  bool store_history = false;
  /// Optional research predicate. Returning true accepts the completed current
  /// iteration and terminates before the next k. Exceptions propagate without
  /// allowing the outer simulator to commit physical state.
  StopPredicate stop_predicate;
};

/// Result of the wave-domain DRS inner iteration in Eq. (17).
struct DrsResult {
  /// Initial auxiliary incident-wave iterate u^{n,0} = P b^n.
  Wave u0;
  /// Last auxiliary iterate u^{n,K_eff}, where K_eff may be smaller than K_n.
  Wave uK;
  /// Last subsystem-response shadow hat{b}^{n,*}; equals b^n when K_n=0.
  Wave b_hat_star;
  /// Coupling-consistent incident wave hat{a}^n = P hat{b}^{n,*} used for advancement.
  Wave a_hat;
  int iterations = 0;
  /// Last ||u^{n,k+1}-u^{n,k}||. This is a DRS fixed-point residual, not state error.
  double last_residual = 0.0;
  DrsTerminationReason termination_reason = DrsTerminationReason::ZeroBudget;
  /// u_history[j] stores u^{n,j}, including the initial iterate.
  std::vector<Wave> u_history;
  /// b_hat_history[j] stores hat{b}^{n,j} = S^n(u^{n,j}).
  std::vector<Wave> b_hat_history;
};

/// Implements the inner loop of Algorithm 1.
///
/// Under FNE of S^n and J_L, the resulting DRS operator is FNE and its auxiliary
/// iterates satisfy the Fejér inequality in Eq. (27). The solver relies on that
/// property but does not attempt to verify the analytical assumptions online.
class DrsInterfaceSolver {
 public:
  DrsResult solve(const StackedFrozenPortMap& S_n,
                  const PreparedWaveInterconnection& interconnection,
                  const Wave& b_previous,
                  const DrsOptions& opts) const {
    if (opts.K_n < 0) {
      throw std::invalid_argument("DrsOptions.K_n must be non-negative");
    }
    if (opts.eps < 0.0) {
      throw std::invalid_argument("DrsOptions.eps must be non-negative");
    }
    // b_previous is the committed outgoing wave b^n at the macro-step boundary.
    require_size(b_previous, interconnection.dimension(), "DRS previous outgoing wave");
    if (S_n.total_port_dimension() != interconnection.dimension()) {
      throw std::invalid_argument("DRS S^n and interconnection dimensions differ");
    }

    DrsResult result;
    // Algorithm 1, line 7:
    // initialize the last response shadow as hat{b}^{n,*}=b^n.
    result.b_hat_star = b_previous;
    // Algorithm 1, line 6:
    // initialize u^{n,0}=P b^n. The same P b^n is the K_n=0 incident wave.
    result.a_hat = interconnection.incident_from_outgoing(b_previous);

    if (opts.K_n == 0) {
      result.u0 = result.a_hat;
      result.uK = result.u0;
      result.termination_reason = DrsTerminationReason::ZeroBudget;
      if (opts.store_history) {
        result.u_history.push_back(result.u0);
      }
      return result;
    }

    // Algorithm 1, line 6:
    // u is the auxiliary incident-wave iterate u^{n,k}, not a committed port input.
    Wave u = interconnection.incident_from_outgoing(b_previous);
    result.u0 = u;
    if (opts.store_history) {
      result.u_history.push_back(u);
    }

    Wave b_hat = b_previous;
    result.termination_reason = DrsTerminationReason::IterationBudget;
    // Algorithm 1, line 9: inner loop.
    for (int k = 0; k < opts.K_n; ++k) {
      // Algorithm 1, line 10:
      // hat{b}^{n,k}=S^n(u^{n,k}). Frozen subsystem states make
      // these evaluations trial responses; no physical state is committed here.
      b_hat = S_n(u);

      // Algorithm 1, line 11:
      // w^{n,k}=J_L(2 hat{b}^{n,k}-u^{n,k}),
      // with L=I-P and J_L=(2I-P)^{-1}; see Eq. (16).
      Wave w = interconnection.resolvent(2.0 * b_hat - u);

      // Algorithm 1, line 12:
      // update u. At a fixed point w=hat{b}, so the increment is zero.
      Wave u_next = u + w - b_hat;

      // Algorithm 1, line 13:
      // retain the most recently evaluated response shadow.
      result.b_hat_star = b_hat;

      // Algorithm 1, lines 14--16:
      // evaluate the optional early-exit test.
      // It measures convergence of the interface iteration, not physical trajectory error.
      result.last_residual = (u_next - u).norm();
      ++result.iterations;
      const bool residual_stop = opts.eps > 0.0 && result.last_residual <= opts.eps;
      bool predicate_stop = false;
      if (opts.stop_predicate) {
        const DrsIterationView view{k, u, u_next, b_hat, w, result.last_residual};
        predicate_stop = opts.stop_predicate(view);
      }
      if (opts.store_history) {
        result.b_hat_history.push_back(b_hat);
        result.u_history.push_back(u_next);
      }
      u = u_next;
      if (residual_stop) {
        result.termination_reason = DrsTerminationReason::ResidualTolerance;
        break;
      }
      if (predicate_stop) {
        result.termination_reason = DrsTerminationReason::UserPredicate;
        break;
      }
    }

    result.uK = u;
    // Algorithm 1, line 19:
    // construct hat{a}^n=P hat{b}^{n,*}. Substituting uK here would be wrong.
    result.a_hat = interconnection.incident_from_outgoing(result.b_hat_star);
    return result;
  }
};

}  // namespace iterative_coupling
