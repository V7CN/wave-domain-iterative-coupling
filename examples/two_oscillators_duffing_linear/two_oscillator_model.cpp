// Generalized pH realization of subsystems A and B for the two-oscillator
// benchmark. The J/R/G/P/S/N matrix entries below are derived term-by-term in
// examples/two_oscillators_duffing_linear/TUTORIAL.md, Sections 4-5.
//
// State order:
//   x_A = (q1, v1, delta),  delta = q2 - q1  (coupling spring extension)
//   x_B = (q2, v2)
// Port causality:
//   A: FlowInputEffortOutput, u_A = v2 (mass-2 velocity), y_A = coupling force
//   B: EffortInputFlowOutput, u_B = coupling force,      y_B = -v2

#include "two_oscillator_model.hpp"

#include "iterative_coupling/integrators/discrete_gradient_phi.hpp"

#include <cmath>

namespace two_oscillator_example {
namespace {

using namespace iterative_coupling;

double energy_A(const PhysicalParameters& p, const State& x) {
  const double q = x(0);
  const double v = x(1);
  const double delta = x(2);
  return 0.5 * p.m1 * v * v + 0.5 * p.k1 * q * q +
         0.25 * p.knl * std::pow(q, 4) + 0.5 * p.k12 * delta * delta;
}

double energy_B(const PhysicalParameters& p, const State& x) {
  return 0.5 * p.m2 * x(1) * x(1) + 0.5 * p.k2 * x(0) * x(0);
}

DiscreteGradientIntegratorOptions integrator_options(const SolverOptions& solver) {
  DiscreteGradientIntegratorOptions options;
  options.tolerance = solver.tolerance;
  options.max_iterations = solver.max_iterations;
  options.max_line_search_iterations = solver.max_line_search_iterations;
  options.armijo_factor = solver.armijo_factor;
  return options;
}

PhiMap make_phi_A(const PhysicalParameters& p, const SolverOptions& solver) {
  Matrix J = Matrix::Zero(3, 3);
  J(0, 1) = 1.0 / p.m1;
  J(1, 0) = -1.0 / p.m1;
  J(1, 2) = 1.0 / p.m1;
  J(2, 1) = -1.0 / p.m1;
  Matrix R = Matrix::Zero(3, 3);
  R(1, 1) = (p.c1 + p.c12) / (p.m1 * p.m1);
  Matrix G = Matrix::Zero(3, 1);
  G(2, 0) = 1.0;
  Matrix P = Matrix::Zero(3, 1);
  P(1, 0) = -p.c12 / p.m1;
  Matrix S = Matrix::Constant(1, 1, p.c12);
  Matrix N = Matrix::Zero(1, 1);

  PortHamiltonianModel model(
      3, 1, [p](const State& x) { return energy_A(p, x); },
      [p](const State& x) {
        Vector gradient(3);
        gradient << p.k1 * x(0) + p.knl * x(0) * x(0) * x(0), p.m1 * x(1),
            p.k12 * x(2);
        return gradient;
      },
      J, R, G, P, S, N, PortCausality::FlowInputEffortOutput);

  DiscreteGradient discrete_gradient(
      3,
      [p](const State& x0, const State& x1) {
        Vector gradient(3);
        const double q0 = x0(0);
        const double q1 = x1(0);
        gradient(0) =
            0.5 * p.k1 * (q0 + q1) +
            0.25 * p.knl *
                (q1 * q1 * q1 + q1 * q1 * q0 + q1 * q0 * q0 + q0 * q0 * q0);
        gradient(1) = 0.5 * p.m1 * (x0(1) + x1(1));
        gradient(2) = 0.5 * p.k12 * (x0(2) + x1(2));
        return gradient;
      },
      [p](const State& x0, const State& x1) {
        Matrix derivative = Matrix::Zero(3, 3);
        const double q0 = x0(0);
        const double q1 = x1(0);
        derivative(0, 0) =
            0.5 * p.k1 +
            0.25 * p.knl * (3.0 * q1 * q1 + 2.0 * q1 * q0 + q0 * q0);
        derivative(1, 1) = 0.5 * p.m1;
        derivative(2, 2) = 0.5 * p.k12;
        return derivative;
      });

  return make_discrete_gradient_phi(model, discrete_gradient, integrator_options(solver));
}

PhiMap make_phi_B(const PhysicalParameters& p, const SolverOptions& solver) {
  Matrix J = Matrix::Zero(2, 2);
  J(0, 1) = 1.0 / p.m2;
  J(1, 0) = -1.0 / p.m2;
  Matrix R = Matrix::Zero(2, 2);
  R(1, 1) = p.c2 / (p.m2 * p.m2);
  Matrix G = Matrix::Zero(2, 1);
  G(1, 0) = -1.0 / p.m2;
  Matrix P = Matrix::Zero(2, 1);
  Matrix S = Matrix::Zero(1, 1);
  Matrix N = Matrix::Zero(1, 1);

  PortHamiltonianModel model(
      2, 1, [p](const State& x) { return energy_B(p, x); },
      [p](const State& x) {
        Vector gradient(2);
        gradient << p.k2 * x(0), p.m2 * x(1);
        return gradient;
      },
      J, R, G, P, S, N, PortCausality::EffortInputFlowOutput);
  Matrix Q = Matrix::Zero(2, 2);
  Q(0, 0) = p.k2;
  Q(1, 1) = p.m2;
  return make_discrete_gradient_phi(model, DiscreteGradient::quadratic_midpoint(Q),
                                    integrator_options(solver));
}

}  // namespace

TwoOscillatorModel make_two_oscillator_model(
    const PhysicalParameters& parameters,
    const InitialConditions& initial,
    const SolverOptions& solver) {
  solver.validate();
  State x_A0(3);
  x_A0 << initial.q1, initial.v1, initial.delta;
  State x_B0(2);
  x_B0 << initial.q2, initial.v2;
  Wave b0(2);
  b0 << initial.bA, initial.bB;
  return TwoOscillatorModel{make_phi_A(parameters, solver),
                            make_phi_B(parameters, solver), std::move(x_A0),
                            std::move(x_B0), std::move(b0)};
}

}  // namespace two_oscillator_example
