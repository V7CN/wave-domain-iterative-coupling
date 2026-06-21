#include "iterative_coupling/core/scattering.hpp"
#include "iterative_coupling/coupling/linear_dirac_interconnection.hpp"
#include "iterative_coupling/solver/drs_interface_solver.hpp"
#include "iterative_coupling/solver/frozen_port_map.hpp"
#include "iterative_coupling/solver/partitioned_simulator.hpp"

#include <cmath>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

using namespace iterative_coupling;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_near(double a, double b, double tol = 1e-12) {
  check(std::abs(a - b) <= tol, "values differ beyond tolerance");
}

PhiMap make_linear_phi(double gain) {
  return PhiMap(
      1, 1,
      [gain](const State& x, const Wave& a, const StepContext&) {
        StepResult r;
        r.x_next = x + a;
        r.b_next = gain * a;
        return r;
      },
      [](const State& x) { return 0.5 * x.squaredNorm(); });
}

void test_scattering_roundtrip_and_power() {
  ScatteringTransform tx(0.4, 0.01);
  PortVariables ef;
  ef.effort = Vector(2);
  ef.flow = Vector(2);
  ef.effort << 3.0, -2.0;
  ef.flow << 0.5, -1.5;
  Wave a = tx.to_incident(ef);
  Wave b = tx.to_outgoing(ef);
  PortVariables recovered = tx.inverse(a, b);
  check((recovered.effort - ef.effort).norm() < 1e-12, "effort round-trip failed");
  check((recovered.flow - ef.flow).norm() < 1e-12, "flow round-trip failed");
  check_near(tx.wave_supply(a, b), ef.effort.dot(ef.flow) * tx.dt(), 1e-12);
}

void test_parallel_executor_real_overlap_and_worker_limits() {
  ParallelExecutor executor(2, 2);
  std::mutex mutex;
  std::condition_variable ready;
  int arrived = 0;
  bool release = false;
  std::set<std::thread::id> thread_ids;
  const auto values = executor.map_indexed(2, [&](std::size_t index) {
    std::unique_lock<std::mutex> lock(mutex);
    thread_ids.insert(std::this_thread::get_id());
    ++arrived;
    if (arrived == 2) {
      release = true;
      ready.notify_all();
    } else if (!ready.wait_for(lock, std::chrono::seconds(2), [&] { return release; })) {
      throw std::runtime_error("parallel tasks did not overlap");
    }
    return static_cast<int>(index);
  });
  check(values == std::vector<int>({0, 1}), "parallel results lost index order");
  check(thread_ids.size() == 2, "parallel work did not use two worker threads");

  ParallelExecutor serial(1, 8);
  ParallelExecutor capped(99, 2);
  check(serial.worker_count() == 1 && !serial.is_parallel(), "serial executor is not serial");
  check(capped.worker_count() == 2, "worker count was not capped by subsystem count");
}

void test_linear_dirac_interconnections() {
  const auto direct = LinearDiracInterconnection::equal_effort_opposite_flow(2);
  const auto prepared_direct = direct.prepare(0.4);
  Matrix expected_swap = Matrix::Zero(4, 4);
  expected_swap.topRightCorner(2, 2) = Matrix::Identity(2, 2);
  expected_swap.bottomLeftCorner(2, 2) = Matrix::Identity(2, 2);
  check((prepared_direct.wave_map() - expected_swap).norm() < 1e-12,
        "paper interconnection did not compile to the swap wave map");

  const auto from_constraints = LinearDiracInterconnection::from_effort_flow_constraints(
      direct.effort_constraints(), direct.flow_constraints());
  check((from_constraints.prepare(0.4).wave_map() - expected_swap).norm() < 1e-12,
        "raw effort/flow constraints differ from the physical factory");

  const auto check_physics = [](const LinearDiracInterconnection& physical, double gamma) {
    const PreparedWaveInterconnection prepared = physical.prepare(gamma);
    Wave outgoing(prepared.dimension());
    for (Eigen::Index i = 0; i < outgoing.size(); ++i) {
      outgoing(i) = 0.17 * static_cast<double>(i + 1) - 0.31;
    }
    const Wave incident = prepared.incident_from_outgoing(outgoing);
    const PortVariables ef = ScatteringTransform(gamma, 0.01).inverse(incident, outgoing);
    check((physical.effort_constraints() * ef.effort +
           physical.flow_constraints() * ef.flow)
                  .norm() <
              1e-11,
          "compiled wave map violates its physical effort/flow constraints");
    check(std::abs(ef.effort.dot(ef.flow)) < 1e-11,
          "linear Dirac interconnection is not lossless");
    check(std::abs(incident.squaredNorm() - outgoing.squaredNorm()) < 1e-11,
          "linear Dirac interconnection does not preserve wave norm");
    check(prepared.diagnostics().constraint_rank == prepared.dimension(),
          "linear Dirac diagnostic reports the wrong rank");
    check(prepared.diagnostics().wave_orthogonality_residual < 1e-12,
          "linear Dirac wave map is not orthogonal");

    Wave value = Wave::LinSpaced(prepared.dimension(), -0.4, 0.7);
    const Wave solution = prepared.resolvent(value);
    check(((2.0 * Matrix::Identity(prepared.dimension(), prepared.dimension()) -
            prepared.wave_map()) *
               solution -
           value)
                  .norm() <
              1e-11,
          "cached interconnection resolvent solve failed");
    check((prepared.resolvent(value) - solution).norm() < 1e-14,
          "repeated cached resolvent calls are inconsistent");
  };

  check_physics(LinearDiracInterconnection::zero_junction(3), 0.7);
  check_physics(LinearDiracInterconnection::one_junction(3), 0.7);
  check_physics(LinearDiracInterconnection::ideal_transformer(2.5), 0.3);
  check_physics(LinearDiracInterconnection::ideal_gyrator(1.7), 1.1);

  const auto recover_port_variables = [](const LinearDiracInterconnection& physical,
                                         double gamma) {
    const PreparedWaveInterconnection prepared = physical.prepare(gamma);
    const Wave outgoing = Wave::LinSpaced(prepared.dimension(), -0.45, 0.35);
    return ScatteringTransform(gamma, 0.01)
        .inverse(prepared.incident_from_outgoing(outgoing), outgoing);
  };
  const PortVariables zero =
      recover_port_variables(LinearDiracInterconnection::zero_junction(3), 0.7);
  check(std::abs(zero.effort(0) - zero.effort(1)) < 1e-12 &&
            std::abs(zero.effort(0) - zero.effort(2)) < 1e-12 &&
            std::abs(zero.flow.sum()) < 1e-12,
        "zero-junction factory has the wrong physical convention");
  const PortVariables one =
      recover_port_variables(LinearDiracInterconnection::one_junction(3), 0.7);
  check(std::abs(one.flow(0) - one.flow(1)) < 1e-12 &&
            std::abs(one.flow(0) - one.flow(2)) < 1e-12 &&
            std::abs(one.effort.sum()) < 1e-12,
        "one-junction factory has the wrong physical convention");
  const PortVariables transformer =
      recover_port_variables(LinearDiracInterconnection::ideal_transformer(2.5), 0.3);
  check(std::abs(transformer.effort(0) - 2.5 * transformer.effort(1)) < 1e-12 &&
            std::abs(2.5 * transformer.flow(0) + transformer.flow(1)) < 1e-12,
        "transformer factory has the wrong physical convention");
  const PortVariables gyrator =
      recover_port_variables(LinearDiracInterconnection::ideal_gyrator(1.7), 1.1);
  check(std::abs(gyrator.effort(0) - 1.7 * gyrator.flow(1)) < 1e-12 &&
            std::abs(gyrator.effort(1) + 1.7 * gyrator.flow(0)) < 1e-12,
        "gyrator factory has the wrong physical convention");

  const PreparedWaveInterconnection shared =
      LinearDiracInterconnection::ideal_gyrator(1.7).prepare(1.1);
  const Wave shared_input = Wave::LinSpaced(shared.dimension(), -0.2, 0.6);
  ParallelExecutor executor(2, 2);
  const auto concurrent = executor.map_indexed(
      2, [&](std::size_t) { return shared.resolvent(shared_input); });
  check((concurrent[0] - concurrent[1]).norm() == 0.0,
        "concurrent cached resolvent calls are inconsistent");

  const auto expect_invalid = [](const std::function<void()>& action) {
    bool rejected = false;
    try {
      action();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "invalid linear Dirac interconnection was not rejected");
  };
  expect_invalid([] {
    (void)LinearDiracInterconnection::from_effort_flow_constraints(Matrix::Identity(2, 2),
                                                                    Matrix::Identity(3, 3));
  });
  expect_invalid([] {
    Matrix E = Matrix::Zero(2, 2);
    Matrix F = Matrix::Zero(2, 2);
    (void)LinearDiracInterconnection::from_effort_flow_constraints(E, F);
  });
  expect_invalid([] {
    Matrix E = Matrix::Identity(2, 2);
    Matrix F = Matrix::Identity(2, 2);
    (void)LinearDiracInterconnection::from_effort_flow_constraints(E, F);
  });
  expect_invalid([] {
    Matrix E = Matrix::Identity(2, 2);
    E(0, 0) = std::numeric_limits<double>::quiet_NaN();
    (void)LinearDiracInterconnection::from_effort_flow_constraints(E, Matrix::Zero(2, 2));
  });
  expect_invalid([] { (void)LinearDiracInterconnection::zero_junction(1); });
  expect_invalid([] { (void)LinearDiracInterconnection::one_junction(3, 0); });
  expect_invalid([] { (void)LinearDiracInterconnection::ideal_transformer(0.0); });
  expect_invalid([] { (void)LinearDiracInterconnection::ideal_gyrator(0.0); });
  expect_invalid([&] { (void)direct.prepare(0.0); });
  expect_invalid([&] { (void)direct.prepare(std::numeric_limits<double>::infinity()); });
}

void test_frozen_port_map_no_mutation() {
  PhiMap phi = make_linear_phi(0.5);
  State x(1);
  x << 2.0;
  StepContext ctx{0.01, 0.4, 0};
  FrozenPortMap S(phi, x, ctx);
  Wave a(1);
  a << 3.0;
  Wave b = S(a);
  check_near(b(0), 1.5);
  check_near(x(0), 2.0);
  StepResult advanced = S.advance(a);
  check_near(advanced.x_next(0), 5.0);
  check_near(x(0), 2.0);
}

void test_drs_k0_and_final_incident() {
  PhiMap phiA = make_linear_phi(0.5);
  PhiMap phiB = make_linear_phi(0.5);
  StepContext ctx{0.01, 0.4, 0};
  State x(1);
  x << 0.0;
  StackedFrozenPortMap S({FrozenPortMap(phiA, x, ctx), FrozenPortMap(phiB, x, ctx)});
  auto coupling = LinearDiracInterconnection::equal_effort_opposite_flow(1).prepare(0.4);
  DrsInterfaceSolver solver;
  Wave b_prev(2);
  b_prev << 1.0, 2.0;

  DrsOptions k0;
  k0.K_n = 0;
  auto r0 = solver.solve(S, coupling, b_prev, k0);
  check((r0.b_hat_star - b_prev).norm() < 1e-12, "K=0 outgoing shadow is wrong");
  check((r0.a_hat - coupling.incident_from_outgoing(b_prev)).norm() < 1e-12,
        "K=0 incident wave is wrong");
  check(r0.termination_reason == DrsTerminationReason::ZeroBudget,
        "K=0 termination reason is wrong");

  DrsOptions k1;
  k1.K_n = 1;
  auto r1 = solver.solve(S, coupling, b_prev, k1);
  Wave expected_bhat(2);
  expected_bhat << 1.0, 0.5;  // 0.5 * (P b_prev)
  check((r1.b_hat_star - expected_bhat).norm() < 1e-12, "DRS shadow response is wrong");
  check((r1.a_hat - coupling.incident_from_outgoing(expected_bhat)).norm() < 1e-12,
        "DRS final incident wave is wrong");
  check((r1.a_hat - r1.uK).norm() > 1e-6, "test did not distinguish a_hat from uK");
  check(r1.termination_reason == DrsTerminationReason::IterationBudget,
        "budget termination reason is wrong");

  DrsOptions early;
  early.K_n = 10;
  early.eps = 1e6;
  auto rearly = solver.solve(S, coupling, b_prev, early);
  check(rearly.iterations == 1, "DRS early-exit did not stop after the first iteration");
  check(rearly.termination_reason == DrsTerminationReason::ResidualTolerance,
        "residual termination reason is wrong");
}

void test_drs_user_stop_predicate() {
  PhiMap phi = make_linear_phi(0.5);
  State x(1);
  x << 0.0;
  StepContext ctx{0.01, 0.4, 0};
  StackedFrozenPortMap map({FrozenPortMap(phi, x, ctx), FrozenPortMap(phi, x, ctx)});
  Wave b_prev(2);
  b_prev << 1.0, 2.0;
  auto coupling = LinearDiracInterconnection::equal_effort_opposite_flow(1).prepare(0.4);

  Wave accepted;
  int calls = 0;
  const std::thread::id caller = std::this_thread::get_id();
  DrsOptions options;
  options.K_n = 10;
  options.stop_predicate = [&](const DrsIterationView& view) {
    check(std::this_thread::get_id() == caller, "stop predicate ran on a worker thread");
    check(view.residual == (view.u_next - view.u_current).norm(),
          "stop predicate received inconsistent residual");
    check(view.b_hat.size() == 2 && view.w.size() == 2,
          "stop predicate received incomplete iteration values");
    ++calls;
    if (view.k == 1) {
      accepted = view.u_next;
      return true;
    }
    return false;
  };
  const DrsResult stopped = DrsInterfaceSolver{}.solve(map, coupling, b_prev, options);
  check(calls == 2 && stopped.iterations == 2, "user predicate stopped at the wrong iteration");
  check(stopped.termination_reason == DrsTerminationReason::UserPredicate,
        "user predicate termination reason is wrong");
  check((stopped.uK - accepted).norm() < 1e-12, "stopped u_next was not accepted as uK");
  check((stopped.a_hat - coupling.incident_from_outgoing(stopped.b_hat_star)).norm() < 1e-12,
        "user stop changed the final incident-wave rule");

  bool predicate_called = false;
  options.eps = 1e6;
  options.stop_predicate = [&](const DrsIterationView&) {
    predicate_called = true;
    return true;
  };
  const DrsResult both = DrsInterfaceSolver{}.solve(map, coupling, b_prev, options);
  check(predicate_called, "predicate was not given the completed residual iteration");
  check(both.termination_reason == DrsTerminationReason::ResidualTolerance,
        "residual tolerance must take precedence when both stop rules trigger");
}

void test_stop_predicate_exception_prevents_commit() {
  PhiMap phi = make_linear_phi(0.5);
  State x(1);
  x << 3.0;
  Wave b0 = Wave::Zero(2);
  DrsOptions options;
  options.K_n = 2;
  options.stop_predicate = [](const DrsIterationView&) -> bool {
    throw std::runtime_error("predicate failure");
  };
  PartitionedSimulator simulator({SubsystemState{phi, x}, SubsystemState{phi, x}}, b0, 0.01,
                                 0.4,
                                 LinearDiracInterconnection::equal_effort_opposite_flow(1), options,
                                 ExecutionOptions{2});
  bool rejected = false;
  try {
    (void)simulator.step(0);
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "predicate failure";
  }
  check(rejected, "stop predicate exception did not propagate");
  check_near(simulator.subsystems()[0].x(0), 3.0);
  check_near(simulator.subsystems()[1].x(0), 3.0);
  check(simulator.logs().steps.empty(), "predicate failure committed a macro-step");
}

void test_simulator_observer_trace_and_single_commit() {
  auto callsA = std::make_shared<int>(0);
  auto callsB = std::make_shared<int>(0);
  auto make_counted_phi = [](const std::shared_ptr<int>& calls) {
    return PhiMap(1, 1, [calls](const State& x, const Wave& a, const StepContext&) {
      ++(*calls);
      StepResult result;
      result.x_next = x + a;
      result.b_next = 0.5 * a;
      result.h_next = 0.5 * result.x_next.squaredNorm();
      return result;
    }, [](const State& x) { return 0.5 * x.squaredNorm(); });
  };

  PhiMap phiA = make_counted_phi(callsA);
  PhiMap phiB = make_counted_phi(callsB);
  State x(1);
  x << 0.0;
  Wave b0(2);
  b0 << 1.0, 2.0;
  const auto physical_interconnection =
      LinearDiracInterconnection::equal_effort_opposite_flow(1);
  auto coupling = physical_interconnection.prepare(0.4);
  DrsOptions opts;
  opts.K_n = 1;

  StepContext ctx{0.01, 0.4, 0};
  StackedFrozenPortMap expected_map({FrozenPortMap(phiA, x, ctx), FrozenPortMap(phiB, x, ctx)});
  DrsResult expected = DrsInterfaceSolver{}.solve(expected_map, coupling, b0, opts);
  *callsA = 0;
  *callsB = 0;

  PartitionedSimulator simulator({SubsystemState{phiA, x}, SubsystemState{phiB, x}}, b0, 0.01,
                                 0.4, physical_interconnection, opts);
  bool observer_called = false;
  StepLog log = simulator.step(
      0, [&](int n, const StackedFrozenPortMap& frozen,
             const PreparedWaveInterconnection&, const Wave& b_prev) {
        check(n == 0, "observer received wrong step index");
        check(*callsA == 0 && *callsB == 0, "observer was not called before finite DRS");
        check((b_prev - b0).norm() < 1e-12, "observer received wrong committed wave");
        Wave probe(1);
        probe << 3.0;
        (void)frozen.map(0)(probe);
        check_near(frozen.map(0).x_frozen()(0), 0.0);
        observer_called = true;
      });

  check(observer_called, "frozen-step observer was not called");
  check((log.u0 - expected.u0).norm() < 1e-12, "StepLog u0 differs from DRS result");
  check((log.uK - expected.uK).norm() < 1e-12, "StepLog uK differs from DRS result");
  check((log.a_used - coupling.incident_from_outgoing(log.b_hat_star)).norm() < 1e-12,
        "simulator did not use P times the final shadow");
  check(*callsA == 3 && *callsB == 2,
        "Algorithm 1 must query each frozen map once and commit each subsystem once");
  check_near(simulator.subsystems()[0].x(0), log.a_used(0));
  check_near(simulator.subsystems()[1].x(0), log.a_used(1));
}

void test_parallel_advance_exception_is_atomic_and_deterministic() {
  auto throwing_phi = [](const char* message) {
    return PhiMap(1, 1,
                   [message](const State&, const Wave&, const StepContext&) -> StepResult {
                     throw std::runtime_error(message);
                   });
  };
  State xA(1);
  State xB(1);
  xA << 1.0;
  xB << 2.0;
  Wave b0 = Wave::Zero(2);
  DrsOptions options;
  options.K_n = 0;
  PartitionedSimulator simulator(
      {{throwing_phi("index-zero"), xA}, {throwing_phi("index-one"), xB}}, b0, 0.01, 0.4,
      LinearDiracInterconnection::equal_effort_opposite_flow(1), options,
      ExecutionOptions{2});
  bool rejected = false;
  try {
    (void)simulator.step(0);
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "index-zero";
  }
  check(rejected, "parallel exceptions were not propagated by lowest subsystem index");
  check_near(simulator.subsystems()[0].x(0), 1.0);
  check_near(simulator.subsystems()[1].x(0), 2.0);
  check(simulator.logs().steps.empty(), "failed parallel macro-step was logged as committed");
}

void test_simulator_gamma_rescatter() {
  PhiMap phi(
      1, 1,
      [](const State& x, const Wave& a, const StepContext& ctx) {
        StepResult r;
        r.x_next = x;
        r.b_next = a;
        r.port_stage = ScatteringTransform(ctx.gamma, ctx.dt).inverse(a, a);
        return r;
      },
      [](const State&) { return 0.0; });
  State x(1);
  x << 0.0;
  Wave b0(2);
  b0 << 0.1, 0.2;
  DrsOptions opts;
  opts.K_n = 0;
  PartitionedSimulator sim({SubsystemState{phi, x}, SubsystemState{phi, x}}, b0, 0.01, 0.4,
                           LinearDiracInterconnection::ideal_gyrator(2.0), opts);
  const Matrix old_wave_map = sim.prepared_interconnection().wave_map();
  const auto log = sim.step(0);
  sim.set_gamma_for_next_step(0.2);
  check((sim.prepared_interconnection().wave_map() - old_wave_map).norm() > 1e-6,
        "gamma change did not rebuild the gyrator wave map");
  ScatteringTransform old_tx(log.gamma_used, 0.01);
  ScatteringTransform new_tx(0.2, 0.01);
  PortVariables ef = old_tx.inverse(log.a_used, log.b_next);
  check((sim.current_outgoing_wave() - new_tx.to_outgoing(ef)).norm() < 1e-12,
        "gamma re-scattering failed");

  const Wave wave_before_failure = sim.current_outgoing_wave();
  const Matrix map_before_failure = sim.prepared_interconnection().wave_map();
  bool rejected = false;
  try {
    sim.set_gamma_for_next_step(std::numeric_limits<double>::quiet_NaN());
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected && sim.gamma() == 0.2,
        "invalid gamma did not fail without changing the simulator gamma");
  check((sim.current_outgoing_wave() - wave_before_failure).norm() == 0.0 &&
            (sim.prepared_interconnection().wave_map() - map_before_failure).norm() == 0.0,
        "failed gamma update changed simulator state");
}

}  // namespace

int main() {
  test_scattering_roundtrip_and_power();
  test_parallel_executor_real_overlap_and_worker_limits();
  test_linear_dirac_interconnections();
  test_frozen_port_map_no_mutation();
  test_drs_k0_and_final_incident();
  test_drs_user_stop_predicate();
  test_stop_predicate_exception_prevents_commit();
  test_simulator_observer_trace_and_single_commit();
  test_parallel_advance_exception_is_atomic_and_deterministic();
  test_simulator_gamma_rescatter();
  std::cout << "All iterative_coupling tests passed\n";
  return 0;
}
