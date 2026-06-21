#pragma once

#include "iterative_coupling/core/types.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace iterative_coupling {

enum class PortCausality {
  /// The model input is flow f and its output is effort e.
  FlowInputEffortOutput,
  /// The model input is effort e and its output is flow f.
  EffortInputFlowOutput,
};

/// Smooth constant-structure port-Hamiltonian model
///   xdot=(J-R) grad H(x)+(G-P)u,
///   y=(G+P)^T grad H(x)+(S+N)u, supply=u^T y=e^T f.
class PortHamiltonianModel {
 public:
  using HamiltonianFunction = std::function<double(const State&)>;
  using GradientFunction = std::function<Vector(const State&)>;

  /// Standard pH form: xdot=(J-R)grad H+G e, f=G^T grad H.
  PortHamiltonianModel(int state_dimension,
                       int port_dimension,
                       HamiltonianFunction hamiltonian,
                       GradientFunction gradient,
                       Matrix J,
                       Matrix R,
                       Matrix G,
                       double structure_tolerance = 1e-12)
      : PortHamiltonianModel(state_dimension, port_dimension, std::move(hamiltonian),
                             std::move(gradient), std::move(J), std::move(R), std::move(G),
                             Matrix::Zero(state_dimension, port_dimension),
                             Matrix::Zero(port_dimension, port_dimension),
                             Matrix::Zero(port_dimension, port_dimension),
                             PortCausality::EffortInputFlowOutput, structure_tolerance) {}

  PortHamiltonianModel(int state_dimension,
                       int port_dimension,
                       HamiltonianFunction hamiltonian,
                       GradientFunction gradient,
                       Matrix J,
                       Matrix R,
                       Matrix G,
                       Matrix P,
                       Matrix S,
                       Matrix N,
                       PortCausality causality,
                       double structure_tolerance = 1e-12)
      : state_dimension_(state_dimension),
        port_dimension_(port_dimension),
        hamiltonian_(std::move(hamiltonian)),
        gradient_(std::move(gradient)),
        J_(std::move(J)),
        R_(std::move(R)),
        G_(std::move(G)),
        P_(std::move(P)),
        S_(std::move(S)),
        N_(std::move(N)),
        causality_(causality),
        structure_tolerance_(structure_tolerance) {
    validate_structure();
  }

  int state_dimension() const { return state_dimension_; }
  int port_dimension() const { return port_dimension_; }
  const Matrix& J() const { return J_; }
  const Matrix& R() const { return R_; }
  const Matrix& G() const { return G_; }
  const Matrix& P() const { return P_; }
  const Matrix& S() const { return S_; }
  const Matrix& N() const { return N_; }
  PortCausality causality() const { return causality_; }

  double hamiltonian(const State& x) const {
    require_size(x, state_dimension_, "PortHamiltonianModel state");
    const double value = hamiltonian_(x);
    if (!std::isfinite(value)) {
      throw std::runtime_error("PortHamiltonianModel Hamiltonian returned a non-finite value");
    }
    return value;
  }

  Vector gradient(const State& x) const {
    require_size(x, state_dimension_, "PortHamiltonianModel gradient state");
    Vector value = gradient_(x);
    require_size(value, state_dimension_, "PortHamiltonianModel gradient result");
    if (!value.allFinite()) {
      throw std::runtime_error("PortHamiltonianModel gradient returned non-finite values");
    }
    return value;
  }

 private:
  void validate_structure() const {
    if (state_dimension_ <= 0 || port_dimension_ <= 0 || structure_tolerance_ < 0.0) {
      throw std::invalid_argument("PortHamiltonianModel dimensions must be positive");
    }
    if (!hamiltonian_ || !gradient_) {
      throw std::invalid_argument("PortHamiltonianModel requires H and grad H callbacks");
    }
    if (J_.rows() != state_dimension_ || J_.cols() != state_dimension_ ||
        R_.rows() != state_dimension_ || R_.cols() != state_dimension_ ||
        G_.rows() != state_dimension_ || G_.cols() != port_dimension_ ||
        P_.rows() != state_dimension_ || P_.cols() != port_dimension_ ||
        S_.rows() != port_dimension_ || S_.cols() != port_dimension_ ||
        N_.rows() != port_dimension_ || N_.cols() != port_dimension_) {
      throw std::invalid_argument("PortHamiltonianModel matrix dimensions are inconsistent");
    }
    if (!J_.allFinite() || !R_.allFinite() || !G_.allFinite() || !P_.allFinite() ||
        !S_.allFinite() || !N_.allFinite()) {
      throw std::invalid_argument("PortHamiltonianModel matrices must be finite");
    }
    const double scale =
        std::max({1.0, J_.norm(), R_.norm(), P_.norm(), S_.norm(), N_.norm()});
    if ((J_ + J_.transpose()).norm() > structure_tolerance_ * scale) {
      throw std::invalid_argument("PortHamiltonianModel J must be skew-symmetric");
    }
    if ((R_ - R_.transpose()).norm() > structure_tolerance_ * scale) {
      throw std::invalid_argument("PortHamiltonianModel R must be symmetric");
    }
    if ((S_ - S_.transpose()).norm() > structure_tolerance_ * scale) {
      throw std::invalid_argument("PortHamiltonianModel S must be symmetric");
    }
    if ((N_ + N_.transpose()).norm() > structure_tolerance_ * scale) {
      throw std::invalid_argument("PortHamiltonianModel N must be skew-symmetric");
    }
    Matrix dissipation = Matrix::Zero(state_dimension_ + port_dimension_,
                                      state_dimension_ + port_dimension_);
    dissipation.topLeftCorner(state_dimension_, state_dimension_) = R_;
    dissipation.topRightCorner(state_dimension_, port_dimension_) = P_;
    dissipation.bottomLeftCorner(port_dimension_, state_dimension_) = P_.transpose();
    dissipation.bottomRightCorner(port_dimension_, port_dimension_) = S_;
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(dissipation);
    if (eigen.info() != Eigen::Success ||
        eigen.eigenvalues().minCoeff() < -structure_tolerance_ * scale) {
      throw std::invalid_argument(
          "PortHamiltonianModel block dissipation matrix [R P; P^T S] must be PSD");
    }
  }

  int state_dimension_;
  int port_dimension_;
  HamiltonianFunction hamiltonian_;
  GradientFunction gradient_;
  Matrix J_;
  Matrix R_;
  Matrix G_;
  Matrix P_;
  Matrix S_;
  Matrix N_;
  PortCausality causality_;
  double structure_tolerance_;
};

}  // namespace iterative_coupling
