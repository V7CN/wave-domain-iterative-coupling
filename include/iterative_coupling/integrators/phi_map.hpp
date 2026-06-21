#pragma once

#include "iterative_coupling/core/types.hpp"

namespace iterative_coupling {

/// Type-erased implementation of the paper's subsystem advancement map
/// Phi_i^{Delta t}: (x_i^n, a_i) -> (x_i^{n+1}, b_i^{n+1}).
///
/// The caller is responsible for supplying a discretization that satisfies the
/// discrete-passivity condition required by the paper. Continuous-time pH form
/// alone does not establish passivity of an arbitrary numerical integrator.
class PhiMap {
 public:
  using StepFunction = std::function<StepResult(const State&, const Wave&, const StepContext&)>;
  using EnergyFunction = std::function<double(const State&)>;

  PhiMap() = delete;

  PhiMap(int state_dimension,
         int port_dimension,
         StepFunction step,
         EnergyFunction energy = EnergyFunction())
      : state_dimension_(state_dimension),
        port_dimension_(port_dimension),
        step_(std::move(step)),
        energy_(std::move(energy)) {
    if (state_dimension_ <= 0 || port_dimension_ <= 0) {
      throw std::invalid_argument("PhiMap dimensions must be positive");
    }
    if (!step_) {
      throw std::invalid_argument("PhiMap requires a step function");
    }
  }

  StepResult step(const State& x_n, const Wave& a_n, const StepContext& ctx) const {
    if (!step_) {
      throw std::runtime_error("PhiMap has no step function");
    }
    require_size(x_n, state_dimension_, "PhiMap state");
    require_size(a_n, port_dimension_, "PhiMap incident wave");
    ctx.validate();
    StepResult result = step_(x_n, a_n, ctx);
    require_size(result.x_next, state_dimension_, "PhiMap result state");
    require_size(result.b_next, port_dimension_, "PhiMap result outgoing wave");
    return result;
  }

  double energy(const State& x) const {
    if (!energy_) {
      throw std::runtime_error("PhiMap has no energy function");
    }
    require_size(x, state_dimension_, "PhiMap energy state");
    return energy_(x);
  }

  bool has_energy() const { return static_cast<bool>(energy_); }
  int state_dimension() const { return state_dimension_; }
  int port_dimension() const { return port_dimension_; }

 private:
  int state_dimension_;
  int port_dimension_;
  StepFunction step_;
  EnergyFunction energy_;
};

}  // namespace iterative_coupling
