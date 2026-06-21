#include "iterative_coupling/integrators/discrete_gradient_phi.hpp"
#include "iterative_coupling/solver/partitioned_simulator.hpp"

#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace iterative_coupling;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_near(double actual, double expected, double tolerance = 1e-10) {
  check(std::abs(actual - expected) <= tolerance, "values differ beyond tolerance");
}

PortHamiltonianModel make_quadratic_model(double damping = 0.0) {
  Matrix J(2, 2);
  J << 0.0, 1.0, -1.0, 0.0;
  Matrix R = Matrix::Zero(2, 2);
  R(1, 1) = damping;
  Matrix G(2, 1);
  G << 0.0, 1.0;
  Matrix Q(2, 2);
  Q << 4.0, 0.0, 0.0, 0.5;
  return PortHamiltonianModel(
      2, 1, [Q](const State& x) { return 0.5 * x.dot(Q * x); },
      [Q](const State& x) { return Q * x; }, J, R, G);
}

void test_standard_constructor_effort_input_semantics() {
  const PortHamiltonianModel model = make_quadratic_model(0.3);
  check(model.causality() == PortCausality::EffortInputFlowOutput,
        "standard J/R/G model does not use effort-input causality");

  const DiscreteGradient dg = DiscreteGradient::gonzalez(model);
  const PhiMap phi = make_discrete_gradient_phi(model, dg);
  State x0(2);
  x0 << 0.2, -0.1;
  Wave a(1);
  a << 0.07;
  const StepContext context{0.01, 0.4, 0};
  const StepResult result = phi.step(x0, a, context);
  const Vector discrete_grad = dg(x0, result.x_next);
  check(result.port_stage.has_value(), "standard J/R/G step omitted physical port variables");
  const PortVariables& port = *result.port_stage;

  const Vector expected_flow = model.G().transpose() * discrete_grad;
  check((port.flow - expected_flow).norm() < 1e-11,
        "standard J/R/G model did not expose G^T discrete_grad as flow");
  const Vector state_residual =
      result.x_next - x0 -
      context.dt * ((model.J() - model.R()) * discrete_grad +
                    model.G() * port.effort);
  check(state_residual.norm() < 1e-11,
        "standard J/R/G model did not use effort as the state-equation input");
  const Vector scattering_residual =
      port.effort + context.gamma * port.flow -
      std::sqrt(2.0 * context.gamma / context.dt) * a;
  check(scattering_residual.norm() < 1e-11,
        "standard J/R/G model used the wrong effort-input scattering equation");
}

void test_model_validation() {
  Matrix J = Matrix::Zero(2, 2);
  Matrix R = Matrix::Zero(2, 2);
  Matrix G = Matrix::Ones(2, 1);
  auto H = [](const State& x) { return 0.5 * x.squaredNorm(); };
  auto grad = [](const State& x) { return x; };
  J(0, 0) = 1.0;
  bool rejected_J = false;
  try {
    (void)PortHamiltonianModel(2, 1, H, grad, J, R, G);
  } catch (const std::invalid_argument&) {
    rejected_J = true;
  }
  check(rejected_J, "non-skew J was accepted");

  J.setZero();
  R(0, 0) = -1.0;
  bool rejected_R = false;
  try {
    (void)PortHamiltonianModel(2, 1, H, grad, J, R, G);
  } catch (const std::invalid_argument&) {
    rejected_R = true;
  }
  check(rejected_R, "indefinite R was accepted");
}

void test_gonzalez_chain_rule_nonlinear() {
  Matrix J = Matrix::Zero(2, 2);
  Matrix R = Matrix::Zero(2, 2);
  Matrix G = Matrix::Identity(2, 2);
  PortHamiltonianModel model(
      2, 2,
      [](const State& x) { return 0.5 * x(0) * x(0) + 0.25 * std::pow(x(1), 4); },
      [](const State& x) {
        Vector gradient(2);
        gradient << x(0), std::pow(x(1), 3);
        return gradient;
      },
      J, R, G);
  const DiscreteGradient dg = DiscreteGradient::gonzalez(model);
  State x0(2);
  State x1(2);
  x0 << -0.3, 0.7;
  x1 << 0.4, -0.2;
  check_near(dg(x0, x1).dot(x1 - x0), model.hamiltonian(x1) - model.hamiltonian(x0),
             1e-13);
}

void test_quadratic_integrator_against_linear_solution_and_analytic_jacobian() {
  const PortHamiltonianModel model = make_quadratic_model(0.3);
  const DiscreteGradient dg = DiscreteGradient::gonzalez(model);
  bool analytic_called = false;
  DiscreteGradientIntegratorOptions options;
  options.residual_jacobian = [&](const State&, const Wave&, const StepContext& context,
                                  const Vector&) {
    analytic_called = true;
    Matrix Q(2, 2);
    Q << 4.0, 0.0, 0.0, 0.5;
    Matrix jacobian = Matrix::Zero(3, 3);
    jacobian.block(0, 0, 2, 2) =
        Matrix::Identity(2, 2) - 0.5 * context.dt * (model.J() - model.R()) * Q;
    jacobian.block(0, 2, 2, 1) = -context.dt * model.G();
    jacobian.block(2, 0, 1, 2) =
        0.5 * context.gamma * model.G().transpose() * Q;
    jacobian(2, 2) = 1.0;
    return jacobian;
  };
  const PhiMap phi = make_discrete_gradient_phi(model, dg, options);
  State x0(2);
  x0 << 0.2, -0.1;
  Wave a(1);
  a << 0.07;
  StepContext context{0.01, 0.4, 0};
  const StepResult result = phi.step(x0, a, context);
  check(analytic_called, "analytic residual Jacobian was not used");

  Matrix Q(2, 2);
  Q << 4.0, 0.0, 0.0, 0.5;
  Matrix system = Matrix::Zero(3, 3);
  system.block(0, 0, 2, 2) =
      Matrix::Identity(2, 2) - 0.5 * context.dt * (model.J() - model.R()) * Q;
  system.block(0, 2, 2, 1) = -context.dt * model.G();
  system.block(2, 0, 1, 2) = 0.5 * context.gamma * model.G().transpose() * Q;
  system(2, 2) = 1.0;
  Vector rhs(3);
  rhs.head(2) =
      (Matrix::Identity(2, 2) + 0.5 * context.dt * (model.J() - model.R()) * Q) * x0;
  rhs(2) = std::sqrt(2.0 * context.gamma / context.dt) * a(0) -
           0.5 * context.gamma * (model.G().transpose() * Q * x0)(0);
  const Vector expected = system.colPivHouseholderQr().solve(rhs);
  check((result.x_next - expected.head(2)).norm() < 1e-11,
        "quadratic discrete-gradient step differs from linear solution");
}

void test_passivity_and_multiport() {
  for (double damping : {0.0, 0.4}) {
    const PortHamiltonianModel model = make_quadratic_model(damping);
    const PhiMap phi = make_discrete_gradient_phi(model, DiscreteGradient::gonzalez(model));
    State x0(2);
    x0 << 0.2, -0.1;
    Wave a(1);
    a << 0.08;
    const StepResult result = phi.step(x0, a, StepContext{0.01, 0.4, 0});
    const double residual = model.hamiltonian(result.x_next) - model.hamiltonian(x0) -
                            0.5 * (a.squaredNorm() - result.b_next.squaredNorm());
    if (damping == 0.0) {
      check_near(residual, 0.0, 2e-11);
    } else {
      check(residual <= 2e-11, "dissipative pH step violated discrete passivity");
    }
  }

  Matrix J(2, 2);
  J << 0.0, 1.0, -1.0, 0.0;
  Matrix R = 0.1 * Matrix::Identity(2, 2);
  Matrix G = Matrix::Identity(2, 2);
  PortHamiltonianModel multiport(
      2, 2, [](const State& x) { return 0.5 * x.squaredNorm(); },
      [](const State& x) { return x; }, J, R, G);
  const PhiMap phi =
      make_discrete_gradient_phi(multiport, DiscreteGradient::gonzalez(multiport));
  State x = State::Zero(2);
  Wave a(2);
  a << 0.1, -0.2;
  check(phi.step(x, a, StepContext{0.02, 0.5, 0}).b_next.size() == 2,
        "multi-port pH step returned the wrong port dimension");
}

void test_generalized_structure_and_both_port_causalities() {
  Matrix J = Matrix::Zero(1, 1);
  Matrix R(1, 1);
  Matrix G(1, 1);
  Matrix P(1, 1);
  Matrix S(1, 1);
  Matrix N = Matrix::Zero(1, 1);
  R(0, 0) = 2.0;
  G(0, 0) = 0.3;
  P(0, 0) = 1.0;
  S(0, 0) = 1.0;
  for (PortCausality causality : {PortCausality::FlowInputEffortOutput,
                                  PortCausality::EffortInputFlowOutput}) {
    PortHamiltonianModel model(
        1, 1, [](const State& x) { return 0.5 * x.squaredNorm(); },
        [](const State& x) { return x; }, J, R, G, P, S, N, causality);
    const PhiMap phi =
        make_discrete_gradient_phi(model, DiscreteGradient::gonzalez(model));
    State x(1);
    x << 0.2;
    Wave a(1);
    a << 0.1;
    const StepResult result = phi.step(x, a, StepContext{0.02, 0.4, 0});
    const double passivity = model.hamiltonian(result.x_next) - model.hamiltonian(x) -
                             0.5 * (a.squaredNorm() - result.b_next.squaredNorm());
    check(passivity <= 2e-11, "generalized pH causality violated passivity");
    check(result.port_stage.has_value(), "generalized pH step omitted physical port values");
  }

  P(0, 0) = 2.0;
  bool rejected = false;
  try {
    (void)PortHamiltonianModel(
        1, 1, [](const State& x) { return 0.5 * x.squaredNorm(); },
        [](const State& x) { return x; }, J, R, G, P, S, N,
        PortCausality::FlowInputEffortOutput);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "non-PSD generalized dissipation block was accepted");
}

void test_affine_fast_path_and_jacobian_precedence() {
  Matrix Q(2, 2);
  Q << 2.0, 0.3, 0.3, 1.5;
  Vector linear_term(2);
  linear_term << 0.1, -0.2;
  Matrix J(2, 2);
  J << 0.0, 0.4, -0.4, 0.0;
  Matrix R = 0.4 * Matrix::Identity(2, 2);
  Matrix G(2, 2);
  G << 1.0, 0.2, -0.1, 0.8;
  Matrix P = 0.05 * Matrix::Identity(2, 2);
  Matrix S = 0.3 * Matrix::Identity(2, 2);
  Matrix N(2, 2);
  N << 0.0, 0.2, -0.2, 0.0;
  State x0(2);
  x0 << 0.25, -0.15;
  Wave a(2);
  a << 0.08, -0.03;
  const StepContext context{0.02, 0.4, 0};

  for (PortCausality causality : {PortCausality::FlowInputEffortOutput,
                                  PortCausality::EffortInputFlowOutput}) {
    PortHamiltonianModel model(
        2, 2,
        [Q, linear_term](const State& x) {
          return 0.5 * x.dot(Q * x) + linear_term.dot(x);
        },
        [Q, linear_term](const State& x) { return (Q * x + linear_term).eval(); },
        J, R, G, P, S, N, causality);
    const DiscreteGradient affine =
        DiscreteGradient::quadratic_midpoint(Q, linear_term);
    const PhiMap direct_phi = make_discrete_gradient_phi(model, affine);
    const StepResult direct = direct_phi.step(x0, a, context);
    check(direct.solver_stats.iterations == 1,
          "quadratic discrete gradient did not use the affine direct path");
    auto concurrent_step = [&direct_phi, &x0, &a, &context]() {
      return direct_phi.step(x0, a, context);
    };
    auto future_a = std::async(std::launch::async, concurrent_step);
    auto future_b = std::async(std::launch::async, concurrent_step);
    check((future_a.get().x_next - direct.x_next).norm() < 1e-14 &&
              (future_b.get().b_next - direct.b_next).norm() < 1e-14,
          "concurrent affine evaluations are not deterministic");

    bool override_called = false;
    DiscreteGradientIntegratorOptions forced_newton;
    forced_newton.residual_jacobian =
        [&, causality](const State&, const Wave&, const StepContext& ctx, const Vector&) {
          override_called = true;
          const Matrix derivative = 0.5 * Q;
          Matrix jacobian = Matrix::Zero(4, 4);
          jacobian.topLeftCorner(2, 2) =
              Matrix::Identity(2, 2) - ctx.dt * (J - R) * derivative;
          jacobian.topRightCorner(2, 2) = -ctx.dt * (G - P);
          if (causality == PortCausality::FlowInputEffortOutput) {
            jacobian.bottomLeftCorner(2, 2) = (G + P).transpose() * derivative;
            jacobian.bottomRightCorner(2, 2) =
                S + N + ctx.gamma * Matrix::Identity(2, 2);
          } else {
            jacobian.bottomLeftCorner(2, 2) =
                ctx.gamma * (G + P).transpose() * derivative;
            jacobian.bottomRightCorner(2, 2) =
                Matrix::Identity(2, 2) + ctx.gamma * (S + N);
          }
          return jacobian;
        };
    const StepResult overridden =
        make_discrete_gradient_phi(model, affine, forced_newton).step(x0, a, context);
    check(override_called, "full residual Jacobian did not override the affine fast path");

    bool discrete_jacobian_called = false;
    const DiscreteGradient analytic(
        2,
        [Q, linear_term](const State& xa, const State& xb) {
          return (0.5 * Q * (xa + xb) + linear_term).eval();
        },
        [&discrete_jacobian_called, Q](const State&, const State&) {
          discrete_jacobian_called = true;
          return 0.5 * Q;
        });
    const StepResult analytic_newton =
        make_discrete_gradient_phi(model, analytic).step(x0, a, context);
    check(discrete_jacobian_called,
          "discrete-gradient next-state Jacobian was not used by Newton");

    const DiscreteGradient finite_difference(
        2, [Q, linear_term](const State& xa, const State& xb) {
          return (0.5 * Q * (xa + xb) + linear_term).eval();
        });
    const StepResult numerical_newton =
        make_discrete_gradient_phi(model, finite_difference).step(x0, a, context);
    for (const StepResult* comparison : {&overridden, &analytic_newton, &numerical_newton}) {
      check((direct.x_next - comparison->x_next).norm() < 1e-12 &&
                (direct.b_next - comparison->b_next).norm() < 1e-12,
            "affine, analytic-Newton, and finite-difference paths disagree");
      check((direct.port_stage->effort - comparison->port_stage->effort).norm() < 1e-12 &&
                (direct.port_stage->flow - comparison->port_stage->flow).norm() < 1e-12,
            "integrator paths recovered different physical port variables");
    }
  }
}

void test_discrete_gradient_capability_validation() {
  bool rejected = false;
  try {
    (void)DiscreteGradient::quadratic_midpoint(Matrix::Ones(2, 3));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "non-square quadratic Q was accepted");

  Matrix nonsymmetric(2, 2);
  nonsymmetric << 1.0, 1.0, 0.0, 1.0;
  rejected = false;
  try {
    (void)DiscreteGradient::quadratic_midpoint(nonsymmetric);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "non-symmetric quadratic Q was accepted");

  DiscreteGradient bad_jacobian(
      2, [](const State&, const State& x1) { return x1; },
      [](const State&, const State&) {
        Matrix bad = Matrix::Zero(2, 2);
        bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
        return bad;
      });
  const PortHamiltonianModel model = make_quadratic_model();
  bool failed = false;
  try {
    State x(2);
    x << 0.2, -0.1;
    (void)make_discrete_gradient_phi(model, bad_jacobian)
        .step(x, Wave::Ones(1), StepContext{0.01, 0.4, 0});
  } catch (const std::runtime_error&) {
    failed = true;
  }
  check(failed, "non-finite discrete-gradient Jacobian was accepted");
}

void test_small_analytic_kernel_matches_generic_residual_override() {
  Matrix Q = Matrix::Zero(3, 3);
  Q.diagonal() << 2.0, 1.5, 0.8;
  Matrix J = Matrix::Zero(3, 3);
  J(0, 1) = 0.4;
  J(1, 0) = -0.4;
  Matrix R = 0.2 * Matrix::Identity(3, 3);
  Matrix G(3, 1);
  G << 0.2, 0.7, -0.1;
  Matrix P = Matrix::Constant(3, 1, 0.02);
  Matrix S = Matrix::Constant(1, 1, 0.1);
  Matrix N = Matrix::Zero(1, 1);
  State x0(3);
  x0 << 0.2, -0.15, 0.05;
  Wave a = Wave::Constant(1, 0.07);
  const StepContext context{0.01, 0.4, 0};
  const DiscreteGradient analytic(
      3, [Q](const State& xa, const State& xb) { return (0.5 * Q * (xa + xb)).eval(); },
      [Q](const State&, const State&) { return 0.5 * Q; });

  for (PortCausality causality : {PortCausality::FlowInputEffortOutput,
                                  PortCausality::EffortInputFlowOutput}) {
    PortHamiltonianModel model(
        3, 1, [Q](const State& x) { return 0.5 * x.dot(Q * x); },
        [Q](const State& x) { return (Q * x).eval(); }, J, R, G, P, S, N, causality);
    const StepResult small = make_discrete_gradient_phi(model, analytic).step(x0, a, context);
    DiscreteGradientIntegratorOptions generic;
    generic.residual_jacobian =
        [=](const State&, const Wave&, const StepContext& ctx, const Vector&) {
          const Matrix derivative = 0.5 * Q;
          Matrix jacobian = Matrix::Zero(4, 4);
          jacobian.topLeftCorner(3, 3) =
              Matrix::Identity(3, 3) - ctx.dt * (J - R) * derivative;
          jacobian.topRightCorner(3, 1) = -ctx.dt * (G - P);
          if (causality == PortCausality::FlowInputEffortOutput) {
            jacobian.bottomLeftCorner(1, 3) = (G + P).transpose() * derivative;
            jacobian(3, 3) = S(0, 0) + ctx.gamma;
          } else {
            jacobian.bottomLeftCorner(1, 3) =
                ctx.gamma * (G + P).transpose() * derivative;
            jacobian(3, 3) = 1.0 + ctx.gamma * S(0, 0);
          }
          return jacobian;
        };
    const StepResult dynamic =
        make_discrete_gradient_phi(model, analytic, generic).step(x0, a, context);
    check((small.x_next - dynamic.x_next).norm() < 1e-13 &&
              (small.b_next - dynamic.b_next).norm() < 1e-13,
          "fixed-size analytic kernel differs from the generic residual path");
  }
}

void test_newton_failure_and_parallel_simulator_equivalence() {
  const PortHamiltonianModel model = make_quadratic_model();
  const DiscreteGradient dg = DiscreteGradient::gonzalez(model);
  DiscreteGradientIntegratorOptions bad;
  bad.residual_jacobian = [](const State&, const Wave&, const StepContext&, const Vector&) {
    return Matrix::Zero(3, 3);
  };
  const PhiMap bad_phi = make_discrete_gradient_phi(model, dg, bad);
  bool rejected = false;
  try {
    (void)bad_phi.step(State::Zero(2), Wave::Ones(1), StepContext{0.01, 0.4, 0});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected, "singular user Jacobian did not fail explicitly");

  const PhiMap phi = make_discrete_gradient_phi(model, dg);
  State xA(2);
  State xB(2);
  xA << 0.1, 0.0;
  xB << 0.0, 0.0;
  Wave b0 = Wave::Zero(2);
  DrsOptions drs;
  drs.K_n = 3;
  PartitionedSimulator serial({SubsystemState{phi, xA}, SubsystemState{phi, xB}}, b0, 0.01,
                              0.4,
                              LinearDiracInterconnection::equal_effort_opposite_flow(1), drs,
                              ExecutionOptions{1});
  PartitionedSimulator parallel({SubsystemState{phi, xA}, SubsystemState{phi, xB}}, b0, 0.01,
                                0.4,
                                LinearDiracInterconnection::equal_effort_opposite_flow(1), drs,
                                ExecutionOptions{2});
  const StepLog serial_log = serial.step(0);
  const StepLog parallel_log = parallel.step(0);
  check((serial_log.b_next - parallel_log.b_next).norm() < 1e-12,
        "parallel generic pH result differs from serial result");
  check((serial_log.x_after[0] - parallel_log.x_after[0]).norm() < 1e-12 &&
            (serial_log.x_after[1] - parallel_log.x_after[1]).norm() < 1e-12,
        "parallel generic pH states differ from serial states");
}

void test_newton_line_search_failure_is_explicit() {
  const PortHamiltonianModel model = make_quadratic_model(0.3);
  const DiscreteGradient dg = DiscreteGradient::gonzalez(model);
  DiscreteGradientIntegratorOptions options;
  options.max_line_search_iterations = 3;
  options.residual_jacobian = [&](const State&, const Wave&, const StepContext& context,
                                  const Vector&) {
    Matrix Q(2, 2);
    Q << 4.0, 0.0, 0.0, 0.5;
    Matrix jacobian = Matrix::Zero(3, 3);
    jacobian.block(0, 0, 2, 2) =
        Matrix::Identity(2, 2) - 0.5 * context.dt * (model.J() - model.R()) * Q;
    jacobian.block(0, 2, 2, 1) = -context.dt * model.G();
    jacobian.block(2, 0, 1, 2) =
        0.5 * context.gamma * model.G().transpose() * Q;
    jacobian(2, 2) = 1.0;
    return (-jacobian).eval();
  };
  State x(2);
  x << 0.2, -0.1;
  bool failed = false;
  try {
    (void)make_discrete_gradient_phi(model, dg, options)
        .step(x, Wave::Constant(1, 0.07), StepContext{0.01, 0.4, 0});
  } catch (const std::runtime_error& error) {
    failed = std::string(error.what()).find("line search") != std::string::npos;
  }
  check(failed, "rejected Newton line search did not fail explicitly");
}

}  // namespace

int main() {
  test_standard_constructor_effort_input_semantics();
  test_model_validation();
  test_gonzalez_chain_rule_nonlinear();
  test_quadratic_integrator_against_linear_solution_and_analytic_jacobian();
  test_passivity_and_multiport();
  test_generalized_structure_and_both_port_causalities();
  test_affine_fast_path_and_jacobian_precedence();
  test_discrete_gradient_capability_validation();
  test_small_analytic_kernel_matches_generic_residual_override();
  test_newton_failure_and_parallel_simulator_equivalence();
  test_newton_line_search_failure_is_explicit();
  std::cout << "All generic port-Hamiltonian tests passed\n";
  return 0;
}
