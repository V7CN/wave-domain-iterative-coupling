#pragma once

#include "iterative_coupling/ph/port_hamiltonian_model.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

namespace iterative_coupling {

class DiscreteGradient {
 public:
  using Function = std::function<Vector(const State&, const State&)>;
  using JacobianFunction = std::function<Matrix(const State&, const State&)>;

  DiscreteGradient(int state_dimension,
                   Function function,
                   JacobianFunction jacobian_x1 = JacobianFunction())
      : state_dimension_(state_dimension),
        function_(std::move(function)),
        jacobian_x1_(std::move(jacobian_x1)) {
    if (state_dimension_ <= 0 || !function_) {
      throw std::invalid_argument("DiscreteGradient requires a positive dimension and callback");
    }
  }

  Vector operator()(const State& x0, const State& x1) const {
    require_size(x0, state_dimension_, "DiscreteGradient x0");
    require_size(x1, state_dimension_, "DiscreteGradient x1");
    Vector value = function_(x0, x1);
    require_size(value, state_dimension_, "DiscreteGradient result");
    if (!value.allFinite()) {
      throw std::runtime_error("DiscreteGradient returned non-finite values");
    }
    return value;
  }

  int state_dimension() const { return state_dimension_; }

  bool has_jacobian_x1() const { return static_cast<bool>(jacobian_x1_); }

  Matrix jacobian_x1(const State& x0, const State& x1) const {
    if (!jacobian_x1_) {
      throw std::runtime_error("DiscreteGradient has no analytic next-state Jacobian");
    }
    require_size(x0, state_dimension_, "DiscreteGradient Jacobian x0");
    require_size(x1, state_dimension_, "DiscreteGradient Jacobian x1");
    Matrix value = jacobian_x1_(x0, x1);
    if (value.rows() != state_dimension_ || value.cols() != state_dimension_ ||
        !value.allFinite()) {
      throw std::runtime_error("DiscreteGradient next-state Jacobian is invalid");
    }
    return value;
  }

  bool has_affine_next_state_representation() const {
    return affine_linear_.rows() == state_dimension_ &&
           affine_linear_.cols() == state_dimension_ && static_cast<bool>(affine_offset_);
  }

  const Matrix& affine_next_state_linear() const {
    if (!has_affine_next_state_representation()) {
      throw std::runtime_error("DiscreteGradient has no affine next-state representation");
    }
    return affine_linear_;
  }

  Vector affine_next_state_offset(const State& x0) const {
    if (!has_affine_next_state_representation()) {
      throw std::runtime_error("DiscreteGradient has no affine next-state representation");
    }
    require_size(x0, state_dimension_, "DiscreteGradient affine x0");
    Vector value = affine_offset_(x0);
    require_size(value, state_dimension_, "DiscreteGradient affine offset");
    if (!value.allFinite()) {
      throw std::runtime_error("DiscreteGradient affine offset is non-finite");
    }
    return value;
  }

  /// Exact midpoint discrete gradient for
  /// H(x)=0.5*x^T*Q*x+linear_term^T*x+constant.
  static DiscreteGradient quadratic_midpoint(Matrix Q,
                                             Vector linear_term = Vector()) {
    if (Q.rows() <= 0 || Q.rows() != Q.cols() || !Q.allFinite()) {
      throw std::invalid_argument("Quadratic discrete gradient requires a finite square Q");
    }
    const int dimension = static_cast<int>(Q.rows());
    if (linear_term.size() == 0) {
      linear_term = Vector::Zero(dimension);
    }
    require_size(linear_term, dimension, "Quadratic discrete-gradient linear term");
    if (!linear_term.allFinite()) {
      throw std::invalid_argument("Quadratic discrete-gradient linear term must be finite");
    }
    const double scale = std::max(1.0, Q.norm());
    if ((Q - Q.transpose()).norm() > 1e-12 * scale) {
      throw std::invalid_argument("Quadratic discrete-gradient Q must be symmetric");
    }

    const Matrix linear = 0.5 * Q;
    DiscreteGradient result(
        dimension,
        [linear, linear_term](const State& x0, const State& x1) {
          return (linear * (x0 + x1) + linear_term).eval();
        },
        [linear](const State&, const State&) { return linear; });
    result.affine_linear_ = linear;
    result.affine_offset_ = [linear, linear_term](const State& x0) {
      return (linear * x0 + linear_term).eval();
    };
    return result;
  }

  static DiscreteGradient gonzalez(const PortHamiltonianModel& model,
                                   double relative_small_step = 1e-14) {
    if (relative_small_step < 0.0) {
      throw std::invalid_argument("Gonzalez small-step tolerance must be non-negative");
    }
    return DiscreteGradient(
        model.state_dimension(),
        [model, relative_small_step](const State& x0, const State& x1) -> Vector {
          const State midpoint = 0.5 * (x0 + x1);
          const Vector gradient_midpoint = model.gradient(midpoint);
          const Vector increment = x1 - x0;
          const double denominator = increment.squaredNorm();
          const double scale = std::max({1.0, x0.squaredNorm(), x1.squaredNorm()});
          if (denominator <= relative_small_step * relative_small_step * scale) {
            return gradient_midpoint;
          }
          const double chain_error = model.hamiltonian(x1) - model.hamiltonian(x0) -
                                     gradient_midpoint.dot(increment);
          return (gradient_midpoint + (chain_error / denominator) * increment).eval();
        });
  }

 private:
  int state_dimension_;
  Function function_;
  JacobianFunction jacobian_x1_;
  Matrix affine_linear_;
  std::function<Vector(const State&)> affine_offset_;
};

}  // namespace iterative_coupling
