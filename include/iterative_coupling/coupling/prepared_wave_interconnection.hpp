#pragma once

#include "iterative_coupling/core/types.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace iterative_coupling {

class LinearDiracInterconnection;

/// Numerical wave-coordinate representation of a validated linear Dirac
/// interconnection. The matrix wave_map_ is the coupling matrix P in the paper;
/// the different C++ name avoids confusion with the P matrix of a generalized
/// port-Hamiltonian subsystem.
struct InterconnectionDiagnostics {
  int constraint_rank = 0;
  double dirac_residual = 0.0;
  double wave_orthogonality_residual = 0.0;
  double resolvent_condition_number = 0.0;
};

class PreparedWaveInterconnection {
 public:
  int dimension() const { return static_cast<int>(wave_map_.rows()); }
  double gamma() const { return gamma_; }

  Wave incident_from_outgoing(const Wave& outgoing) const {
    require_size(outgoing, wave_map_.cols(), "Interconnection outgoing wave");
    return wave_map_ * outgoing;
  }

  Wave resolvent(const Wave& value) const {
    require_size(value, wave_map_.rows(), "Interconnection resolvent input");
    return resolvent_qr_.solve(value);
  }

  const Matrix& wave_map() const { return wave_map_; }
  const InterconnectionDiagnostics& diagnostics() const { return diagnostics_; }

 private:
  friend class LinearDiracInterconnection;

  PreparedWaveInterconnection(Matrix wave_map,
                              double gamma,
                              int constraint_rank,
                              double dirac_residual,
                              double tolerance)
      : wave_map_(std::move(wave_map)), gamma_(gamma) {
    const Matrix identity = Matrix::Identity(wave_map_.rows(), wave_map_.cols());
    diagnostics_.constraint_rank = constraint_rank;
    diagnostics_.dirac_residual = dirac_residual;
    diagnostics_.wave_orthogonality_residual =
        (wave_map_.transpose() * wave_map_ - identity).norm() /
        std::max(1.0, identity.norm());
    if (!wave_map_.allFinite() ||
        diagnostics_.wave_orthogonality_residual > tolerance) {
      throw std::invalid_argument(
          "Compiled Dirac interconnection is not an orthogonal wave map; residual=" +
          std::to_string(diagnostics_.wave_orthogonality_residual));
    }

    const Matrix resolvent_system = 2.0 * identity - wave_map_;
    resolvent_qr_.compute(resolvent_system);
    if (!resolvent_qr_.isInvertible()) {
      throw std::invalid_argument("Compiled interconnection resolvent is singular");
    }
    const Eigen::JacobiSVD<Matrix> svd(resolvent_system);
    const auto singular_values = svd.singularValues();
    const double smallest = singular_values.minCoeff();
    if (!(smallest > 0.0) || !std::isfinite(smallest)) {
      throw std::invalid_argument("Compiled interconnection resolvent is ill-defined");
    }
    diagnostics_.resolvent_condition_number = singular_values.maxCoeff() / smallest;
  }

  Matrix wave_map_;
  double gamma_;
  Eigen::ColPivHouseholderQR<Matrix> resolvent_qr_;
  InterconnectionDiagnostics diagnostics_;
};

}  // namespace iterative_coupling
