#include "iterative_coupling/coupling/linear_dirac_interconnection.hpp"
#include "iterative_coupling/integrators/discrete_gradient_phi.hpp"
#include "iterative_coupling/solver/partitioned_simulator.hpp"

#include <iostream>

using namespace iterative_coupling;

int main() {
  const double mass = 2.0;
  const double stiffness = 30.0;
  const double damping = 0.2;
  Matrix J(2, 2);
  J << 0.0, 1.0, -1.0, 0.0;
  Matrix R = Matrix::Zero(2, 2);
  R(1, 1) = damping;
  Matrix G(2, 1);
  G << 0.0, 1.0;

  PortHamiltonianModel model(
      2, 1,
      [=](const State& x) {
        return 0.5 * stiffness * x(0) * x(0) + 0.5 * x(1) * x(1) / mass;
      },
      [=](const State& x) {
        Vector gradient(2);
        gradient << stiffness * x(0), x(1) / mass;
        return gradient;
      },
      J, R, G);
  const PhiMap phi = make_discrete_gradient_phi(model, DiscreteGradient::gonzalez(model));

  State xA(2);
  State xB(2);
  xA << 0.1, 0.0;
  xB << 0.0, 0.0;
  Wave b0 = Wave::Zero(2);
  DrsOptions drs;
  drs.K_n = 8;
  PartitionedSimulator simulator(
      {SubsystemState{phi, xA}, SubsystemState{phi, xB}}, b0, 0.01, 0.4,
      LinearDiracInterconnection::equal_effort_opposite_flow(1), drs);
  simulator.run(100);
  std::cout << "Generic pH example final states:\n"
            << simulator.subsystems()[0].x.transpose() << "\n"
            << simulator.subsystems()[1].x.transpose() << "\n";
  return 0;
}
