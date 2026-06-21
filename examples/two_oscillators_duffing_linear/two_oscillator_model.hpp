#pragma once

#include "iterative_coupling/integrators/phi_map.hpp"

#include <stdexcept>

namespace two_oscillator_example {

inline constexpr const char* kModelProvenance = "generalized-ph";

struct PhysicalParameters {
  double m1 = 8.0;
  double k1 = 100.0;
  double c1 = 0.0;
  double m2 = 4.0;
  double k2 = 50.0;
  double c2 = 0.0;
  double k12 = 120.0;
  double c12 = 0.05;
  double knl = 8000.0;
};

struct InitialConditions {
  double q1 = 0.4;
  double v1 = 0.0;
  double delta = -0.4;
  double q2 = 0.0;
  double v2 = 0.0;
  double bA = 0.0;
  double bB = 0.0;
};

struct SolverOptions {
  double tolerance = 1e-12;
  int max_iterations = 25;
  int max_line_search_iterations = 10;
  double armijo_factor = 1e-4;

  void validate() const {
    if (tolerance <= 0.0 || max_iterations <= 0 || max_line_search_iterations <= 0 ||
        armijo_factor <= 0.0) {
      throw std::invalid_argument("Two-oscillator solver options must be positive");
    }
  }
};

struct TwoOscillatorModel {
  iterative_coupling::PhiMap phi_A;
  iterative_coupling::PhiMap phi_B;
  iterative_coupling::State x_A0;
  iterative_coupling::State x_B0;
  iterative_coupling::Wave b0;
};

TwoOscillatorModel make_two_oscillator_model(
    const PhysicalParameters& parameters,
    const InitialConditions& initial,
    const SolverOptions& solver);

}  // namespace two_oscillator_example
