#pragma once

#include "iterative_coupling/coupling/prepared_wave_interconnection.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace iterative_coupling {

/// Constant maximal linear Dirac interconnection
///
///     E e + F f = 0,
///
/// in stacked physical effort/flow coordinates. Maximality and losslessness
/// are enforced by rank([E F])=m and E F^T + F E^T=0. Port blocks follow the
/// same order as the subsystem vector passed to PartitionedSimulator.
class LinearDiracInterconnection {
 public:
  static LinearDiracInterconnection from_effort_flow_constraints(
      Matrix E, Matrix F, double tolerance = 1e-10) {
    return LinearDiracInterconnection(std::move(E), std::move(F), tolerance);
  }

  static LinearDiracInterconnection equal_effort_opposite_flow(
      int port_dimension = 1) {
    require_positive(port_dimension, "port_dimension");
    const int total = 2 * port_dimension;
    Matrix E = Matrix::Zero(total, total);
    Matrix F = Matrix::Zero(total, total);
    const Matrix identity = Matrix::Identity(port_dimension, port_dimension);
    E.block(0, 0, port_dimension, port_dimension) = identity;
    E.block(0, port_dimension, port_dimension, port_dimension) = -identity;
    F.block(port_dimension, 0, port_dimension, port_dimension) = identity;
    F.block(port_dimension, port_dimension, port_dimension, port_dimension) = identity;
    return from_effort_flow_constraints(std::move(E), std::move(F));
  }

  static LinearDiracInterconnection zero_junction(int port_count,
                                                   int port_dimension = 1) {
    require_junction_dimensions(port_count, port_dimension);
    const int total = port_count * port_dimension;
    Matrix E = Matrix::Zero(total, total);
    Matrix F = Matrix::Zero(total, total);
    const Matrix identity = Matrix::Identity(port_dimension, port_dimension);
    for (int port = 1; port < port_count; ++port) {
      const int row = (port - 1) * port_dimension;
      E.block(row, 0, port_dimension, port_dimension) = identity;
      E.block(row, port * port_dimension, port_dimension, port_dimension) = -identity;
    }
    const int final_row = (port_count - 1) * port_dimension;
    for (int port = 0; port < port_count; ++port) {
      F.block(final_row, port * port_dimension, port_dimension, port_dimension) = identity;
    }
    return from_effort_flow_constraints(std::move(E), std::move(F));
  }

  static LinearDiracInterconnection one_junction(int port_count,
                                                  int port_dimension = 1) {
    require_junction_dimensions(port_count, port_dimension);
    const int total = port_count * port_dimension;
    Matrix E = Matrix::Zero(total, total);
    Matrix F = Matrix::Zero(total, total);
    const Matrix identity = Matrix::Identity(port_dimension, port_dimension);
    for (int port = 1; port < port_count; ++port) {
      const int row = (port - 1) * port_dimension;
      F.block(row, 0, port_dimension, port_dimension) = identity;
      F.block(row, port * port_dimension, port_dimension, port_dimension) = -identity;
    }
    const int final_row = (port_count - 1) * port_dimension;
    for (int port = 0; port < port_count; ++port) {
      E.block(final_row, port * port_dimension, port_dimension, port_dimension) = identity;
    }
    return from_effort_flow_constraints(std::move(E), std::move(F));
  }

  static LinearDiracInterconnection ideal_transformer(double ratio,
                                                       int port_dimension = 1) {
    require_nonzero_finite(ratio, "transformer ratio");
    require_positive(port_dimension, "port_dimension");
    const int total = 2 * port_dimension;
    Matrix E = Matrix::Zero(total, total);
    Matrix F = Matrix::Zero(total, total);
    const Matrix identity = Matrix::Identity(port_dimension, port_dimension);
    E.block(0, 0, port_dimension, port_dimension) = identity;
    E.block(0, port_dimension, port_dimension, port_dimension) = -ratio * identity;
    F.block(port_dimension, 0, port_dimension, port_dimension) = ratio * identity;
    F.block(port_dimension, port_dimension, port_dimension, port_dimension) = identity;
    return from_effort_flow_constraints(std::move(E), std::move(F));
  }

  static LinearDiracInterconnection ideal_gyrator(double gyration,
                                                   int port_dimension = 1) {
    require_nonzero_finite(gyration, "gyration");
    require_positive(port_dimension, "port_dimension");
    const int total = 2 * port_dimension;
    Matrix E = Matrix::Identity(total, total);
    Matrix F = Matrix::Zero(total, total);
    const Matrix identity = Matrix::Identity(port_dimension, port_dimension);
    F.block(0, port_dimension, port_dimension, port_dimension) = -gyration * identity;
    F.block(port_dimension, 0, port_dimension, port_dimension) = gyration * identity;
    return from_effort_flow_constraints(std::move(E), std::move(F));
  }

  int dimension() const { return static_cast<int>(E_.rows()); }
  const Matrix& effort_constraints() const { return E_; }
  const Matrix& flow_constraints() const { return F_; }

  PreparedWaveInterconnection prepare(double gamma) const {
    if (!std::isfinite(gamma) || gamma <= 0.0) {
      throw std::invalid_argument("Interconnection gamma must be finite and positive");
    }
    const double root_gamma = std::sqrt(gamma);
    const Matrix M_a = root_gamma * E_ + F_ / root_gamma;
    const Matrix M_b = root_gamma * E_ - F_ / root_gamma;
    Eigen::ColPivHouseholderQR<Matrix> qr(M_a);
    qr.setThreshold(tolerance_);
    if (!qr.isInvertible()) {
      throw std::invalid_argument("Dirac interconnection incident-wave system is singular");
    }
    Matrix wave_map = qr.solve(-M_b);
    return PreparedWaveInterconnection(std::move(wave_map), gamma, constraint_rank_,
                                       dirac_residual_, tolerance_);
  }

 private:
  LinearDiracInterconnection(Matrix E, Matrix F, double tolerance)
      : E_(std::move(E)), F_(std::move(F)), tolerance_(tolerance) {
    if (!std::isfinite(tolerance_) || tolerance_ < 0.0) {
      throw std::invalid_argument(
          "Dirac interconnection tolerance must be finite and non-negative");
    }
    if (E_.rows() <= 0 || E_.rows() != E_.cols() || F_.rows() != E_.rows() ||
        F_.cols() != E_.cols()) {
      throw std::invalid_argument(
          "Dirac interconnection E and F must be equal non-empty square matrices");
    }
    if (!E_.allFinite() || !F_.allFinite()) {
      throw std::invalid_argument("Dirac interconnection matrices must be finite");
    }

    Matrix constraints(E_.rows(), 2 * E_.cols());
    constraints << E_, F_;
    Eigen::FullPivLU<Matrix> rank_decomposition(constraints);
    rank_decomposition.setThreshold(tolerance_);
    constraint_rank_ = static_cast<int>(rank_decomposition.rank());
    if (constraint_rank_ != dimension()) {
      throw std::invalid_argument("Dirac interconnection [E F] must have full row rank");
    }

    const Matrix dirac_identity = E_ * F_.transpose() + F_ * E_.transpose();
    const double scale = std::max(1.0, E_.norm() * F_.norm());
    dirac_residual_ = dirac_identity.norm() / scale;
    if (dirac_residual_ > tolerance_) {
      throw std::invalid_argument(
          "Dirac interconnection violates E F^T + F E^T = 0; residual=" +
          std::to_string(dirac_residual_));
    }
  }

  static void require_positive(int value, const char* name) {
    if (value <= 0) {
      throw std::invalid_argument(std::string(name) + " must be positive");
    }
  }

  static void require_junction_dimensions(int port_count, int port_dimension) {
    if (port_count < 2) {
      throw std::invalid_argument("junction port_count must be at least two");
    }
    require_positive(port_dimension, "port_dimension");
  }

  static void require_nonzero_finite(double value, const char* name) {
    if (!std::isfinite(value) || value == 0.0) {
      throw std::invalid_argument(std::string(name) + " must be finite and non-zero");
    }
  }

  Matrix E_;
  Matrix F_;
  double tolerance_;
  int constraint_rank_ = 0;
  double dirac_residual_ = 0.0;
};

}  // namespace iterative_coupling
