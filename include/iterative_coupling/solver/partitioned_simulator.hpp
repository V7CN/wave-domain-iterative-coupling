#pragma once

#include "iterative_coupling/core/scattering.hpp"
#include "iterative_coupling/coupling/linear_dirac_interconnection.hpp"
#include "iterative_coupling/solver/drs_interface_solver.hpp"

#include <functional>
#include <memory>

namespace iterative_coupling {

struct SubsystemState {
  PhiMap phi;
  State x;
};

/// Executes the outer macro-step loop of Algorithm 1.
///
/// The simulator owns the committed states x_i^n and outgoing wave b^n. Each
/// call to step() freezes those states, solves the interface problem, evaluates
/// every Phi_i^{Delta t} once more at hat{a}_i^n, and only then commits n+1.
class PartitionedSimulator {
 public:
  /// Synchronous read-only hook invoked after S^n is frozen and before the
  /// finite-budget DRS solve. References passed to the hook are valid only for
  /// the duration of the call and must not be retained. Evaluating the frozen
  /// maps is side-effect free and cannot commit physical subsystem state.
  using FrozenStepObserver = std::function<void(int,
                                                const StackedFrozenPortMap&,
                                                const PreparedWaveInterconnection&,
                                                const Wave&)>;

  PartitionedSimulator(std::vector<SubsystemState> subsystems,
                       Wave b_initial,
                       double dt,
                       double gamma,
                       LinearDiracInterconnection interconnection,
                       DrsOptions drs_options,
                       ExecutionOptions execution_options = {})
      : subsystems_(std::move(subsystems)),
        b_current_(std::move(b_initial)),
        dt_(dt),
        gamma_(gamma),
        interconnection_(std::move(interconnection)),
        prepared_interconnection_(std::make_unique<PreparedWaveInterconnection>(
            interconnection_.prepare(gamma_))),
        drs_options_(drs_options) {
    if (subsystems_.empty()) {
      throw std::invalid_argument("PartitionedSimulator requires at least one subsystem");
    }
    if (!std::isfinite(dt_) || dt_ <= 0.0 || !std::isfinite(gamma_) || gamma_ <= 0.0) {
      throw std::invalid_argument("PartitionedSimulator dt/gamma must be finite and positive");
    }
    total_port_dimension_ = 0;
    for (const auto& subsystem : subsystems_) {
      require_size(subsystem.x, subsystem.phi.state_dimension(), "Subsystem initial state");
      total_port_dimension_ += subsystem.phi.port_dimension();
    }
    require_size(b_current_, total_port_dimension_, "Initial outgoing wave");
    if (interconnection_.dimension() != total_port_dimension_) {
      throw std::invalid_argument("Interconnection dimension does not match subsystem ports");
    }
    executor_ = std::make_unique<ParallelExecutor>(execution_options.worker_count,
                                                   subsystems_.size());
  }

  StepLog step(int step_index, const FrozenStepObserver& observer = {}) {
    // Algorithm 1, line 1:
    // execute one outer macro-step n using the currently committed state.
    // Snapshot Delta t and gamma for macro-step n. This context is shared by all
    // frozen-map queries and by the final committed subsystem advancement.
    StepContext ctx{dt_, gamma_, step_index};
    ctx.validate();

    std::vector<FrozenPortMap> frozen;
    frozen.reserve(subsystems_.size());

    StepLog log;
    log.step_index = step_index;
    log.time = static_cast<double>(step_index) * dt_;
    log.gamma_used = gamma_;
    log.K_n = drs_options_.K_n;
    // Algorithm 1, line 5:
    // b^n=col(b_1^n,...,b_N^n) is the committed outgoing wave.
    log.b_prev = b_current_;

    // Algorithm 1, line 3:
    // freeze each committed x_i^n to form S^n=diag(S_1^n,...,S_N^n).
    // Inner evaluations of these maps cannot alter subsystems_[i].x.
    for (const auto& subsystem : subsystems_) {
      log.x_before.push_back(subsystem.x);
      frozen.emplace_back(subsystem.phi, subsystem.x, ctx);
    }

    StackedFrozenPortMap S_n(std::move(frozen), executor_.get());
    if (observer) {
      // This hook is for offline experiment diagnostics (FNE sampling and a
      // long-DRS reference). It is not a step of Algorithm 1 and runs before
      // the finite-budget online solve on the same frozen S^n.
      observer(step_index, S_n, *prepared_interconnection_, b_current_);
    }
    // Solve the finite-budget interface problem and obtain
    // hat{a}^n=P hat{b}^{n,*}. No physical time advancement occurs inside solve().
    DrsResult drs =
        drs_solver_.solve(S_n, *prepared_interconnection_, b_current_, drs_options_);
    log.inner_iterations = drs.iterations;
    log.inner_residual = drs.last_residual;
    log.termination_reason = drs.termination_reason;
    log.u0 = drs.u0;
    log.uK = drs.uK;
    log.b_hat_star = drs.b_hat_star;
    log.a_used = drs.a_hat;

    Wave b_next(total_port_dimension_);
    log.energy.clear();
    std::vector<Wave> incident;
    incident.reserve(subsystems_.size());
    Eigen::Index incident_offset = 0;
    for (std::size_t i = 0; i < subsystems_.size(); ++i) {
      const int m = subsystems_[i].phi.port_dimension();
      // Extract hat{a}_i^n from the stacked coupling-consistent incident wave.
      incident.push_back(drs.a_hat.segment(incident_offset, m));
      incident_offset += m;
    }
    const std::vector<StepResult> advanced_results = executor_->map_indexed(
        subsystems_.size(), [&S_n, &incident](std::size_t i) {
          return S_n.map(i).advance(incident[i]);
        });

    Eigen::Index offset = 0;
    for (std::size_t i = 0; i < subsystems_.size(); ++i) {
      const int m = subsystems_[i].phi.port_dimension();
      const Wave& a_i = incident[i];
      double h_before = subsystems_[i].phi.has_energy()
                            ? subsystems_[i].phi.energy(subsystems_[i].x)
                            : 0.0;
      // Algorithm 1, lines 20--21:
      // final macro-step advancement from the same frozen state:
      // (x_i^{n+1},b_i^{n+1})=Phi_i^{Delta t}(x_i^n,hat{a}_i^n).
      // Unlike an inner S_i^n query, this result is eligible for commitment.
      const StepResult& advanced = advanced_results[i];
      double h_after = advanced.h_next.value_or(
          subsystems_[i].phi.has_energy() ? subsystems_[i].phi.energy(advanced.x_next) : 0.0);
      EnergyDiagnostics ed;
      ed.h_before = h_before;
      ed.h_after = h_after;
      ed.supply = 0.5 * (a_i.squaredNorm() - advanced.b_next.squaredNorm());
      // r_i^n=Delta H_i-0.5(||hat{a}_i^n||^2-||b_i^{n+1}||^2).
      // Condition 2 requires r_i^n<=0.
      ed.passivity_residual = (h_after - h_before) - ed.supply;
      log.energy.push_back(ed);
      log.subsystem_solver_stats.push_back(advanced.solver_stats);

      // This is the only write to committed physical state during macro-step n;
      // all S_i^n evaluations above discarded their trial states.
      subsystems_[i].x = advanced.x_next;
      log.x_after.push_back(subsystems_[i].x);
      b_next.segment(offset, m) = advanced.b_next;
      offset += m;
    }

    b_current_ = b_next;
    log.b_next = b_current_;
    last_a_used_ = log.a_used;
    last_b_next_ = log.b_next;
    last_gamma_used_ = log.gamma_used;
    have_last_step_ = true;
    logs_.steps.push_back(log);
    return log;
  }

  SimulationLog run(int steps) {
    if (steps < 0) {
      throw std::invalid_argument("run steps must be non-negative");
    }
    for (int n = 0; n < steps; ++n) {
      step(n);
    }
    return logs_;
  }

  void set_gamma_for_next_step(double new_gamma) {
    if (!std::isfinite(new_gamma) || new_gamma <= 0.0) {
      throw std::invalid_argument("new gamma must be finite and positive");
    }
    if (new_gamma == gamma_) {
      return;
    }
    auto new_interconnection = std::make_unique<PreparedWaveInterconnection>(
        interconnection_.prepare(new_gamma));
    Wave new_outgoing = b_current_;
    if (have_last_step_) {
      // A wave value cannot be reinterpreted under a new gamma. Recover the
      // physical (e,f) pair with gamma_n, then scatter it again with gamma_{n+1}.
      const ScatteringTransform old_transform(last_gamma_used_, dt_);
      const PortVariables ef = old_transform.inverse(last_a_used_, last_b_next_);
      const ScatteringTransform new_transform(new_gamma, dt_);
      new_outgoing = new_transform.to_outgoing(ef);
    }
    prepared_interconnection_.swap(new_interconnection);
    b_current_.swap(new_outgoing);
    gamma_ = new_gamma;
  }

  const std::vector<SubsystemState>& subsystems() const { return subsystems_; }
  const Wave& current_outgoing_wave() const { return b_current_; }
  double gamma() const { return gamma_; }
  const PreparedWaveInterconnection& prepared_interconnection() const {
    return *prepared_interconnection_;
  }
  std::size_t worker_count() const { return executor_->worker_count(); }
  const SimulationLog& logs() const { return logs_; }

 private:
  std::vector<SubsystemState> subsystems_;
  Wave b_current_;
  double dt_;
  double gamma_;
  int total_port_dimension_ = 0;
  LinearDiracInterconnection interconnection_;
  std::unique_ptr<PreparedWaveInterconnection> prepared_interconnection_;
  DrsOptions drs_options_;
  DrsInterfaceSolver drs_solver_;
  std::unique_ptr<ParallelExecutor> executor_;
  SimulationLog logs_;

  bool have_last_step_ = false;
  Wave last_a_used_;
  Wave last_b_next_;
  double last_gamma_used_ = 0.0;
};

}  // namespace iterative_coupling
