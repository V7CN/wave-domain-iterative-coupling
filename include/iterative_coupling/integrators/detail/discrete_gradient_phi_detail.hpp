#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace iterative_coupling::detail {

inline Vector solve_small_dense(const Matrix& matrix,
                                const Vector& rhs,
                                const char* singular_message) {
  if (matrix.rows() != matrix.cols() || matrix.rows() != rhs.size() ||
      !matrix.allFinite() || !rhs.allFinite()) {
    throw std::runtime_error("Discrete-gradient linear system is invalid");
  }
  if (matrix.rows() == 3) {
    const Eigen::Matrix3d fixed_matrix = matrix;
    const Eigen::Vector3d fixed_rhs = rhs;
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> factorization(fixed_matrix);
    if (!factorization.isInvertible()) {
      throw std::runtime_error(singular_message);
    }
    return factorization.solve(fixed_rhs);
  }
  if (matrix.rows() == 4) {
    const Eigen::Matrix4d fixed_matrix = matrix;
    const Eigen::Vector4d fixed_rhs = rhs;
    Eigen::ColPivHouseholderQR<Eigen::Matrix4d> factorization(fixed_matrix);
    if (!factorization.isInvertible()) {
      throw std::runtime_error(singular_message);
    }
    return factorization.solve(fixed_rhs);
  }
  Eigen::ColPivHouseholderQR<Matrix> factorization(matrix);
  if (!factorization.isInvertible()) {
    throw std::runtime_error(singular_message);
  }
  return factorization.solve(rhs);
}

class PreparedDiscreteGradientIntegrator {
 public:
  PreparedDiscreteGradientIntegrator(PortHamiltonianModel model,
                                     DiscreteGradient discrete_gradient,
                                     DiscreteGradientIntegratorOptions options)
      : model_(std::move(model)),
        discrete_gradient_(std::move(discrete_gradient)),
        options_(std::move(options)),
        n_(model_.state_dimension()),
        m_(model_.port_dimension()),
        dynamics_(model_.J() - model_.R()),
        input_matrix_(model_.G() - model_.P()),
        output_transpose_((model_.G() + model_.P()).transpose()),
        feedthrough_(model_.S() + model_.N()),
        identity_state_(Matrix::Identity(n_, n_)),
        identity_port_(Matrix::Identity(m_, m_)) {
    if (n_ == 3 && m_ == 1) {
      dynamics_3_ = dynamics_;
      input_3_ = input_matrix_.col(0);
      output_3_ = output_transpose_.row(0).transpose();
      feedthrough_1_ = feedthrough_(0, 0);
    }
  }

  StepResult step(const State& x0, const Wave& a, const StepContext& context) const {
    const Vector incident_rhs = std::sqrt(2.0 * context.gamma / context.dt) * a;
    if (discrete_gradient_.has_affine_next_state_representation() &&
        !options_.residual_jacobian) {
      return affine_step(x0, incident_rhs, context);
    }
    return newton_step(x0, a, incident_rhs, context);
  }

 private:
  Vector residual(const State& x0,
                  const Vector& value,
                  const Vector& incident_rhs,
                  const StepContext& context) const {
    const State x1 = value.head(n_);
    const Vector discrete_grad = discrete_gradient_(x0, x1);
    Vector result(n_ + m_);
    if (m_ == 1) {
      const double input = value(n_);
      const double output = output_transpose_.row(0).dot(discrete_grad) +
                            feedthrough_(0, 0) * input;
      result.head(n_).noalias() =
          x1 - x0 -
          context.dt * (dynamics_ * discrete_grad + input_matrix_.col(0) * input);
      result(n_) = model_.causality() == PortCausality::FlowInputEffortOutput
                       ? output + context.gamma * input - incident_rhs(0)
                       : input + context.gamma * output - incident_rhs(0);
      return result;
    }
    const Vector input = value.tail(m_);
    const Vector output = output_transpose_ * discrete_grad + feedthrough_ * input;
    result.head(n_).noalias() =
        x1 - x0 - context.dt * (dynamics_ * discrete_grad + input_matrix_ * input);
    if (model_.causality() == PortCausality::FlowInputEffortOutput) {
      result.tail(m_).noalias() =
          output + context.gamma * input - incident_rhs;
    } else {
      result.tail(m_).noalias() =
          input + context.gamma * output - incident_rhs;
    }
    return result;
  }

  Matrix analytic_jacobian(const Matrix& discrete_gradient_jacobian,
                           const StepContext& context) const {
    Matrix jacobian = Matrix::Zero(n_ + m_, n_ + m_);
    jacobian.topLeftCorner(n_, n_).noalias() =
        identity_state_ - context.dt * dynamics_ * discrete_gradient_jacobian;
    jacobian.topRightCorner(n_, m_).noalias() = -context.dt * input_matrix_;
    if (model_.causality() == PortCausality::FlowInputEffortOutput) {
      jacobian.bottomLeftCorner(m_, n_).noalias() =
          output_transpose_ * discrete_gradient_jacobian;
      jacobian.bottomRightCorner(m_, m_) =
          feedthrough_ + context.gamma * identity_port_;
    } else {
      jacobian.bottomLeftCorner(m_, n_).noalias() =
          context.gamma * output_transpose_ * discrete_gradient_jacobian;
      jacobian.bottomRightCorner(m_, m_) =
          identity_port_ + context.gamma * feedthrough_;
    }
    return jacobian;
  }

  Matrix finite_difference_jacobian(const State& x0,
                                    const Vector& value,
                                    const Vector& incident_rhs,
                                    const StepContext& context) const {
    Matrix jacobian(n_ + m_, n_ + m_);
    for (int column = 0; column < n_ + m_; ++column) {
      const double step = options_.finite_difference_relative_step *
                          std::max(1.0, std::abs(value(column)));
      Vector plus = value;
      Vector minus = value;
      plus(column) += step;
      minus(column) -= step;
      jacobian.col(column) =
          (residual(x0, plus, incident_rhs, context) -
           residual(x0, minus, incident_rhs, context)) /
          (2.0 * step);
    }
    return jacobian;
  }

  Vector initial_input(const Matrix& matrix,
                       const Vector& rhs,
                       const char* singular_message) const {
    if (m_ == 1) {
      if (std::abs(matrix(0, 0)) <=
          1e-15 * std::max(1.0, std::abs(rhs(0)))) {
        throw std::runtime_error(singular_message);
      }
      return Vector::Constant(1, rhs(0) / matrix(0, 0));
    }
    return solve_small_dense(matrix, rhs, singular_message);
  }

  StepResult affine_step(const State& x0,
                         const Vector& incident_rhs,
                         const StepContext& context) const {
    const Matrix& linear = discrete_gradient_.affine_next_state_linear();
    const Vector offset = discrete_gradient_.affine_next_state_offset(x0);
    Matrix system = analytic_jacobian(linear, context);
    Vector rhs(n_ + m_);
    rhs.head(n_).noalias() = x0 + context.dt * dynamics_ * offset;
    if (model_.causality() == PortCausality::FlowInputEffortOutput) {
      rhs.tail(m_).noalias() = incident_rhs - output_transpose_ * offset;
    } else {
      rhs.tail(m_).noalias() =
          incident_rhs - context.gamma * output_transpose_ * offset;
    }
    const Vector solution =
        solve_small_dense(system, rhs, "Discrete-gradient affine system is singular");
    const double residual_norm = (system * solution - rhs).norm();
    if (residual_norm > options_.tolerance) {
      throw std::runtime_error("Discrete-gradient affine solve did not satisfy tolerance");
    }
    const Vector discrete_grad =
        linear * solution.head(n_) + offset;
    return make_result_from_gradient(solution, discrete_grad, context, 1, residual_norm);
  }

  StepResult newton_step(const State& x0,
                         const Wave& a,
                         const Vector& incident_rhs,
                         const StepContext& context) const {
    if (n_ == 3 && m_ == 1 && !options_.residual_jacobian &&
        discrete_gradient_.has_jacobian_x1()) {
      return analytic_newton_step_3_1(x0, incident_rhs(0), context);
    }
    const Vector grad0 = model_.gradient(x0);
    Matrix port_matrix;
    Vector port_rhs;
    if (model_.causality() == PortCausality::FlowInputEffortOutput) {
      port_matrix = feedthrough_ + context.gamma * identity_port_;
      port_rhs = incident_rhs - output_transpose_ * grad0;
    } else {
      port_matrix = identity_port_ + context.gamma * feedthrough_;
      port_rhs = incident_rhs - context.gamma * output_transpose_ * grad0;
    }
    const Vector input0 = initial_input(
        port_matrix, port_rhs, "Discrete-gradient initial port system is singular");
    Vector z(n_ + m_);
    z.head(n_).noalias() =
        x0 + context.dt * (dynamics_ * grad0 + input_matrix_ * input0);
    z.tail(m_) = input0;

    int iterations = 0;
    bool converged = false;
    double residual_norm = 0.0;
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
      iterations = iteration + 1;
      const Vector current_residual = residual(x0, z, incident_rhs, context);
      residual_norm = current_residual.norm();
      if (residual_norm <= options_.tolerance) {
        converged = true;
        break;
      }

      Matrix jacobian;
      if (options_.residual_jacobian) {
        jacobian = options_.residual_jacobian(x0, a, context, z);
      } else if (discrete_gradient_.has_jacobian_x1()) {
        jacobian = analytic_jacobian(
            discrete_gradient_.jacobian_x1(x0, z.head(n_)), context);
      } else {
        jacobian = finite_difference_jacobian(x0, z, incident_rhs, context);
      }
      if (jacobian.rows() != n_ + m_ || jacobian.cols() != n_ + m_ ||
          !jacobian.allFinite()) {
        throw std::runtime_error("Discrete-gradient residual Jacobian is invalid");
      }
      const Vector update = solve_small_dense(
          jacobian, -current_residual, "Discrete-gradient Newton Jacobian is singular");

      double step_length = 1.0;
      bool accepted = false;
      for (int line_search = 0; line_search < options_.max_line_search_iterations;
           ++line_search) {
        const Vector candidate = z + step_length * update;
        if (residual(x0, candidate, incident_rhs, context).norm() <=
            (1.0 - options_.armijo_factor * step_length) * residual_norm) {
          z = candidate;
          accepted = true;
          break;
        }
        step_length *= 0.5;
      }
      if (!accepted) {
        throw std::runtime_error("Discrete-gradient Newton line search failed");
      }
    }

    residual_norm = residual(x0, z, incident_rhs, context).norm();
    converged = converged || residual_norm <= options_.tolerance;
    if (!converged) {
      throw std::runtime_error("Discrete-gradient Newton did not converge");
    }
    return make_result(x0, z, context, iterations, residual_norm);
  }

  Eigen::Vector4d residual_3_1(const State& x0,
                               const Eigen::Vector4d& value,
                               double incident_rhs,
                               const StepContext& context) const {
    const State x1 = value.head<3>();
    const Eigen::Vector3d discrete_grad = discrete_gradient_(x0, x1);
    const double input = value(3);
    const double output = output_3_.dot(discrete_grad) + feedthrough_1_ * input;
    Eigen::Vector4d result;
    result.head<3>() = value.head<3>() - x0.head<3>() -
                       context.dt *
                           (dynamics_3_ * discrete_grad + input_3_ * input);
    result(3) = model_.causality() == PortCausality::FlowInputEffortOutput
                    ? output + context.gamma * input - incident_rhs
                    : input + context.gamma * output - incident_rhs;
    return result;
  }

  Eigen::Matrix4d analytic_jacobian_3_1(const Matrix& derivative,
                                         const StepContext& context) const {
    const Eigen::Matrix3d fixed_derivative = derivative;
    Eigen::Matrix4d jacobian = Eigen::Matrix4d::Zero();
    jacobian.topLeftCorner<3, 3>() =
        Eigen::Matrix3d::Identity() - context.dt * dynamics_3_ * fixed_derivative;
    jacobian.topRightCorner<3, 1>() = -context.dt * input_3_;
    if (model_.causality() == PortCausality::FlowInputEffortOutput) {
      jacobian.bottomLeftCorner<1, 3>() = output_3_.transpose() * fixed_derivative;
      jacobian(3, 3) = feedthrough_1_ + context.gamma;
    } else {
      jacobian.bottomLeftCorner<1, 3>() =
          context.gamma * output_3_.transpose() * fixed_derivative;
      jacobian(3, 3) = 1.0 + context.gamma * feedthrough_1_;
    }
    return jacobian;
  }

  StepResult analytic_newton_step_3_1(const State& x0,
                                      double incident_rhs,
                                      const StepContext& context) const {
    const Eigen::Vector3d grad0 = model_.gradient(x0);
    const double port_matrix =
        model_.causality() == PortCausality::FlowInputEffortOutput
            ? feedthrough_1_ + context.gamma
            : 1.0 + context.gamma * feedthrough_1_;
    const double port_rhs =
        model_.causality() == PortCausality::FlowInputEffortOutput
            ? incident_rhs - output_3_.dot(grad0)
            : incident_rhs - context.gamma * output_3_.dot(grad0);
    if (std::abs(port_matrix) <= 1e-15 * std::max(1.0, std::abs(port_rhs))) {
      throw std::runtime_error("Discrete-gradient initial port system is singular");
    }
    const double input0 = port_rhs / port_matrix;
    Eigen::Vector4d z;
    z.head<3>() = x0.head<3>() +
                  context.dt * (dynamics_3_ * grad0 + input_3_ * input0);
    z(3) = input0;

    int iterations = 0;
    bool converged = false;
    double residual_norm = 0.0;
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
      iterations = iteration + 1;
      const Eigen::Vector4d current_residual =
          residual_3_1(x0, z, incident_rhs, context);
      residual_norm = current_residual.norm();
      if (residual_norm <= options_.tolerance) {
        converged = true;
        break;
      }
      const State x1 = z.head<3>();
      const Eigen::Matrix4d jacobian = analytic_jacobian_3_1(
          discrete_gradient_.jacobian_x1(x0, x1), context);
      Eigen::ColPivHouseholderQR<Eigen::Matrix4d> factorization(jacobian);
      if (!factorization.isInvertible()) {
        throw std::runtime_error("Discrete-gradient Newton Jacobian is singular");
      }
      const Eigen::Vector4d update = factorization.solve(-current_residual);

      double step_length = 1.0;
      bool accepted = false;
      for (int line_search = 0; line_search < options_.max_line_search_iterations;
           ++line_search) {
        const Eigen::Vector4d candidate = z + step_length * update;
        if (residual_3_1(x0, candidate, incident_rhs, context).norm() <=
            (1.0 - options_.armijo_factor * step_length) * residual_norm) {
          z = candidate;
          accepted = true;
          break;
        }
        step_length *= 0.5;
      }
      if (!accepted) {
        throw std::runtime_error("Discrete-gradient Newton line search failed");
      }
    }

    residual_norm = residual_3_1(x0, z, incident_rhs, context).norm();
    converged = converged || residual_norm <= options_.tolerance;
    if (!converged) {
      throw std::runtime_error("Discrete-gradient Newton did not converge");
    }
    const Vector solution = z;
    return make_result(x0, solution, context, iterations, residual_norm);
  }

  StepResult make_result(const State& x0,
                         const Vector& solution,
                         const StepContext& context,
                         int iterations,
                         double residual_norm) const {
    const State x1 = solution.head(n_);
    const Vector discrete_grad = discrete_gradient_(x0, x1);
    return make_result_from_gradient(solution, discrete_grad, context, iterations,
                                     residual_norm);
  }

  StepResult make_result_from_gradient(const Vector& solution,
                                       const Vector& discrete_grad,
                                       const StepContext& context,
                                       int iterations,
                                       double residual_norm) const {
    const State x1 = solution.head(n_);
    const Vector input = solution.tail(m_);
    const Vector output = output_transpose_ * discrete_grad + feedthrough_ * input;
    const Vector flow = model_.causality() == PortCausality::FlowInputEffortOutput
                            ? input
                            : output;
    const Vector effort = model_.causality() == PortCausality::FlowInputEffortOutput
                              ? output
                              : input;
    StepResult result;
    result.x_next = x1;
    result.b_next = std::sqrt(context.dt) * (effort - context.gamma * flow) /
                    std::sqrt(2.0 * context.gamma);
    result.port_stage = PortVariables{effort, flow};
    result.h_next = model_.hamiltonian(x1);
    result.solver_stats = SolverStats{iterations, residual_norm};
    return result;
  }

  PortHamiltonianModel model_;
  DiscreteGradient discrete_gradient_;
  DiscreteGradientIntegratorOptions options_;
  int n_;
  int m_;
  Matrix dynamics_;
  Matrix input_matrix_;
  Matrix output_transpose_;
  Matrix feedthrough_;
  Matrix identity_state_;
  Matrix identity_port_;
  Eigen::Matrix3d dynamics_3_ = Eigen::Matrix3d::Zero();
  Eigen::Vector3d input_3_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d output_3_ = Eigen::Vector3d::Zero();
  double feedthrough_1_ = 0.0;
};

}  // namespace iterative_coupling::detail
