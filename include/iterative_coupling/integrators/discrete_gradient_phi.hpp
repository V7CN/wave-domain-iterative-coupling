#pragma once

#include "iterative_coupling/integrators/discrete_gradient.hpp"
#include "iterative_coupling/integrators/phi_map.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

namespace iterative_coupling {

struct DiscreteGradientIntegratorOptions {
  using ResidualJacobian =
      std::function<Matrix(const State&, const Wave&, const StepContext&, const Vector&)>;

  double tolerance = 1e-12;
  int max_iterations = 30;
  int max_line_search_iterations = 12;
  double armijo_factor = 1e-4;
  double finite_difference_relative_step = 1e-7;
  ResidualJacobian residual_jacobian;

  void validate() const {
    if (tolerance <= 0.0 || max_iterations <= 0 || max_line_search_iterations <= 0 ||
        armijo_factor <= 0.0 || finite_difference_relative_step <= 0.0) {
      throw std::invalid_argument("Discrete-gradient integrator options must be positive");
    }
  }
};

}  // namespace iterative_coupling

#include "iterative_coupling/integrators/detail/discrete_gradient_phi_detail.hpp"

namespace iterative_coupling {

inline PhiMap make_discrete_gradient_phi(
    const PortHamiltonianModel& model,
    const DiscreteGradient& discrete_gradient,
    const DiscreteGradientIntegratorOptions& options = {}) {
  options.validate();
  if (discrete_gradient.state_dimension() != model.state_dimension()) {
    throw std::invalid_argument("Discrete gradient and pH model dimensions differ");
  }
  const detail::PreparedDiscreteGradientIntegrator prepared(model, discrete_gradient, options);
  return PhiMap(
      model.state_dimension(), model.port_dimension(),
      [prepared](const State& x0, const Wave& a, const StepContext& context) {
        return prepared.step(x0, a, context);
      },
      [model](const State& x) { return model.hamiltonian(x); });
}

}  // namespace iterative_coupling
