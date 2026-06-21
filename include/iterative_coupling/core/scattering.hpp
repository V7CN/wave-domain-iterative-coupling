#pragma once

#include "iterative_coupling/core/types.hpp"

#include <cmath>

namespace iterative_coupling {

class ScatteringTransform {
 public:
  ScatteringTransform(double gamma, double dt) : gamma_(gamma), dt_(dt) {
    if (gamma_ <= 0.0) {
      throw std::invalid_argument("ScatteringTransform gamma must be positive");
    }
    if (dt_ <= 0.0) {
      throw std::invalid_argument("ScatteringTransform dt must be positive");
    }
  }

  double gamma() const { return gamma_; }
  double dt() const { return dt_; }

  Wave to_incident(const PortVariables& ef) const {
    if (ef.effort.size() != ef.flow.size()) {
      throw std::invalid_argument("PortVariables effort/flow dimensions differ");
    }
    return std::sqrt(dt_) * (ef.effort + gamma_ * ef.flow) / std::sqrt(2.0 * gamma_);
  }

  Wave to_outgoing(const PortVariables& ef) const {
    if (ef.effort.size() != ef.flow.size()) {
      throw std::invalid_argument("PortVariables effort/flow dimensions differ");
    }
    return std::sqrt(dt_) * (ef.effort - gamma_ * ef.flow) / std::sqrt(2.0 * gamma_);
  }

  PortVariables inverse(const Wave& a, const Wave& b) const {
    if (a.size() != b.size()) {
      throw std::invalid_argument("Incident/outgoing wave dimensions differ");
    }
    PortVariables ef;
    ef.effort = std::sqrt(gamma_ / (2.0 * dt_)) * (a + b);
    ef.flow = (a - b) / std::sqrt(2.0 * gamma_ * dt_);
    return ef;
  }

  double wave_supply(const Wave& a, const Wave& b) const {
    if (a.size() != b.size()) {
      throw std::invalid_argument("Incident/outgoing wave dimensions differ");
    }
    return 0.5 * (a.squaredNorm() - b.squaredNorm());
  }

 private:
  double gamma_;
  double dt_;
};

}  // namespace iterative_coupling
