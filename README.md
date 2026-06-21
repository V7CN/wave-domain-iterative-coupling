# Wave-Domain Iterative Coupling

[![arXiv](https://img.shields.io/badge/arXiv-2603.16424-b31b1b.svg)](https://arxiv.org/abs/2603.16424)

Implementation of the wave-domain Douglas-Rachford splitting coupling interface described in

> **Early-Terminable Energy-Safe Iterative Coupling for Parallel Simulation of Port-Hamiltonian Systems**
> Qi Wei, Jianfeng Tao, Hongyu Nie, Wangtao Tan
> arXiv:2603.16424, 2026
> [arXiv:2603.16424](https://arxiv.org/abs/2603.16424)

The method partitions a port-Hamiltonian system into subsystems, exchanges power through wave (scattering) variables, and reconciles the lossless coupling constraint via a Douglas-Rachford inner iteration. For any finite inner-iteration budget `K_n`, the macro-step satisfies the augmented-storage passivity inequality (Theorem 1). As `K_n` grows, the partitioned trajectory converges to the monolithic discrete-gradient trajectory (Theorem 2).

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/two_oscillators_benchmark
./build/generic_ph_mass_spring
```

Requires Eigen3 and C++17.

```bash
# macOS
brew install eigen

# Ubuntu / Debian
sudo apt install libeigen3-dev
```

The core library depends only on Eigen3 and the C++ standard library.

The benchmark writes trajectory CSVs, diagnostic CSVs, and `summary.json` to `examples/two_oscillators_duffing_linear/out_cpp/` (override with the first positional argument). See [Outputs and reproduction](#outputs-and-reproduction) for the CSV-to-figure mapping. `generic_ph_mass_spring` prints final states to stdout and is a minimal starting point.

## Repository layout

```text
include/iterative_coupling/
  core/         Wave, State, ScatteringTransform, StepContext, ParallelExecutor
  coupling/     LinearDiracInterconnection, PreparedWaveInterconnection
  integrators/  PhiMap contract, DiscreteGradient, discrete-gradient adapter
  ph/           PortHamiltonianModel (generalized pH form)
  solver/       FrozenPortMap, StackedFrozenPortMap, DrsInterfaceSolver,
                PartitionedSimulator
examples/
  two_oscillators_duffing_linear/  Paper benchmark (Section V). See TUTORIAL.md.
  generic_ph_mass_spring/          Minimal two-subsystem example.
tests/        core_tests, ph_tests, benchmark_tests
paper/        LaTeX source of the paper.
```

## Documentation

- **[TUTORIAL.md](examples/two_oscillators_duffing_linear/TUTORIAL.md)** — step-by-step walkthrough of the two-oscillator benchmark, from the physical sketch through generalized-pH matrices to diagnostic CSV columns.
- **[ADDING_A_PH_EXAMPLE.md](ADDING_A_PH_EXAMPLE.md)** — recipe for adding a new port-Hamiltonian example: define subsystems, choose a discrete gradient, set up the interconnection, and configure Algorithm 1.

## Paper-to-code map

| PDF object | Source location |
|---|---|
| Eqs. (3), (4), (5) scattering | [core/scattering.hpp](include/iterative_coupling/core/scattering.hpp) |
| Eq. (7) lossless coupling, Eq. (9) wave matrix | [coupling/linear_dirac_interconnection.hpp](include/iterative_coupling/coupling/linear_dirac_interconnection.hpp) |
| Eq. (12) frozen port map | [solver/frozen_port_map.hpp](include/iterative_coupling/solver/frozen_port_map.hpp) |
| Eqs. (16), (17) DRS operators and inner iteration | [solver/drs_interface_solver.hpp](include/iterative_coupling/solver/drs_interface_solver.hpp) |
| Algorithm 1 outer macro-step | [solver/partitioned_simulator.hpp::step](include/iterative_coupling/solver/partitioned_simulator.hpp#L67) |
| Algorithm 1 inner DRS loop | [solver/drs_interface_solver.hpp::solve](include/iterative_coupling/solver/drs_interface_solver.hpp#L61) |
| Eq. (26) per-subsystem discrete-passivity residual | [solver/partitioned_simulator.hpp:147-150](include/iterative_coupling/solver/partitioned_simulator.hpp#L147-L150) |
| Condition 1 sampling (offline FNE margins) | [examples/.../benchmark_diagnostics.cpp::fne_sampling_check](examples/two_oscillators_duffing_linear/benchmark_diagnostics.cpp#L17) |
| Theorem 1 augmented-storage residual (offline) | [examples/.../benchmark_diagnostics.cpp::augmented_storage_residual](examples/two_oscillators_duffing_linear/benchmark_diagnostics.cpp#L49) |
| Two-oscillator benchmark parameters (Table I) | [examples/.../benchmark_config.hpp:23-32](examples/two_oscillators_duffing_linear/benchmark_config.hpp#L23-L32) |

## Core abstractions

| Abstraction | Responsibility |
|---|---|
| [PhiMap](include/iterative_coupling/integrators/phi_map.hpp) | Subsystem advancement $\Phi_i^{\Delta t}(x_i^n, a_i) \mapsto (x_i^{n+1}, b_i^{n+1})$. Supplied by the user; must satisfy Condition 2. |
| [PortHamiltonianModel](include/iterative_coupling/ph/port_hamiltonian_model.hpp) + [DiscreteGradient](include/iterative_coupling/integrators/discrete_gradient.hpp) | Constant-structure generalized pH model plus a structure-preserving discrete gradient. `make_discrete_gradient_phi` adapts the pair into a `PhiMap`. |
| [LinearDiracInterconnection](include/iterative_coupling/coupling/linear_dirac_interconnection.hpp) + [PreparedWaveInterconnection](include/iterative_coupling/coupling/prepared_wave_interconnection.hpp) | Physical `E e + F f = 0` relation, compiled to the orthogonal wave map `a = P b` and the cached DRS resolvent `(2I - P)^{-1}` for a given `gamma`. |
| [PartitionedSimulator](include/iterative_coupling/solver/partitioned_simulator.hpp) | Owns the committed states and outgoing wave. Each `step(n)` runs Algorithm 1 once. |

**Two different matrices called $P$.** The paper uses $P$ for the external wave coupling matrix (Eq. 9). The generalized-pH subsystem also has a $P$ matrix in its dissipative input-output structure. The C++ code keeps the names distinct: `wave_map_` (or `U` in comments) for the external coupling matrix, `P_` for the subsystem matrix. When a markdown section below says $P$ without qualification, it refers to the subsystem matrix.

## Subsystems: port-Hamiltonian models

The library convention is

$$
\dot x = (J - R)\, \nabla H(x) + (G - P)\, u, \qquad
y = (G + P)^\top \nabla H(x) + (S + N)\, u, \qquad \text{supplied power} = u^\top y.
$$

`PortCausality::FlowInputEffortOutput` identifies $(u, y) = (f, e)$; `EffortInputFlowOutput` identifies $(u, y) = (e, f)$. The constructor validates skew $J/N$, symmetric $R/S$, and positive-semidefinite $\begin{pmatrix} R & P \\ P^\top & S \end{pmatrix}$.

**Choosing the port causality.** The port input $u$ is the physical quantity the interconnection delivers to the subsystem at each macro-step. Pick by the physical nature of that quantity:

| If the port input is a … | Example domains | Use |
|---|---|---|
| flow-like quantity (velocity, angular velocity, volumetric flow, current) | mechanical translation/rotation, hydraulic, electrical current source | `FlowInputEffortOutput` |
| effort-like quantity (force, torque, pressure, voltage) | force source, pressure source, voltage source | `EffortInputFlowOutput` |

Both subsystems in a coupled pair must use compatible causalities so that the lossless constraint $e_A = e_B$, $f_A + f_B = 0$ closes. The two-oscillator benchmark mixes them: subsystem A receives the velocity of mass 2 (flow input), subsystem B receives the coupling force (effort input). See [TUTORIAL.md §4-5](examples/two_oscillators_duffing_linear/TUTORIAL.md) for the detailed construction.

Construct `PortHamiltonianModel` from `H`, `grad H`, and constant `J/R/G`, then adapt it to Algorithm 1:

```cpp
PortHamiltonianModel model(n, m, H, gradH, J, R, G);
DiscreteGradient dg = DiscreteGradient::gonzalez(model);
PhiMap phi = make_discrete_gradient_phi(model, dg);
```

The paper's Eq. (1), $\dot x=(J-R)\nabla H+Ge$ and $f=G^\top\nabla H$, is the special case $P=S=N=0$ and $(u,y)=(e,f)$ (`EffortInputFlowOutput`) of this generalized model. The `J/R/G` constructor represents that standard effort-input form. Nonzero $P/S/N$, flow-input causality, and general maximal linear Dirac interconnections are reusable library extensions beyond Eq. (1).

The adapter solves for `x_next` and the stage input selected by the port causality, recovers the conjugate stage output and outgoing scaled wave, and checks Newton convergence. The default Jacobian is centered finite difference; the user may supply the full residual Jacobian. User-defined passive steppers can construct `PhiMap` directly.

`DiscreteGradient` may optionally provide its next-state Jacobian. The adapter assembles the generalized-pH residual Jacobian itself, so model code does not repeat causality or scattering blocks. Quadratic Hamiltonians can use `DiscreteGradient::quadratic_midpoint(Q, linear_term)`: this marks the residual as affine and selects a single direct solve. These capabilities are optional; the minimal construction above remains unchanged.

Scope: smooth Hamiltonians, constant $J/R/G/P/S/N$, constant maximal linear Dirac interconnections, one scalar `gamma`, and synchronous macro-steps. State-dependent or nonlinear interconnections, dissipative networks, unequal port impedances, and asynchronous communication are not supported. See [ADDING_A_PH_EXAMPLE.md](ADDING_A_PH_EXAMPLE.md) for the model-to-simulation workflow.

## Interconnections

The paper's coupling (Eq. 7) is

$$
e_A = e_B, \qquad f_A + f_B = 0.
$$

In the wave domain this becomes the orthogonal map $a = P b$ (Eq. 9). The code accepts the paper's constraint directly:

```cpp
auto interconnection =
    LinearDiracInterconnection::equal_effort_opposite_flow(port_dimension);
```

This constraint is a special case of a **linear Dirac structure** — a maximal, power-conserving relation between port efforts and flows:

$$
E\, e + F\, f = 0, \qquad
\operatorname{rank}(\begin{pmatrix} E & F \end{pmatrix}) = m, \qquad
E F^\top + F E^\top = 0.
$$

The paper uses the two-port swap matrix $P$ for readability, but every mathematical result depends only on $P$ being orthogonal. A general linear Dirac structure compiles to an orthogonal $P$, so Theorems 1 and 2 carry over unchanged. The code chooses the Dirac form because defining an interconnection in physical effort/flow coordinates is more intuitive and less error-prone than hand-constructing an orthogonal $P$ directly. `LinearDiracInterconnection` validates the Dirac identities, compiles the physical relation to the orthogonal wave map $a = U b$, and caches the DRS resolvent $(2I - U)^{-1}$.

Factories for bond-graph 0/1 junctions, ideal transformers, and ideal gyrators are included. General constant lossless networks use `from_effort_flow_constraints(E, F)`. Port blocks in `E` and `F` follow the subsystem order passed to `PartitionedSimulator`.

`gamma` may change only between macro-steps. `set_gamma_for_next_step()` re-scatters the committed wave and rebuilds the prepared wave map; a macro-step always uses one fixed value.

## Parallel execution

`PartitionedSimulator` uses a persistent C++17 thread pool. By default the worker count is `min(hardware_concurrency, subsystem_count)`; pass `ExecutionOptions{1}` for deterministic serial execution. Frozen-map queries and final subsystem advances run concurrently; results and exceptions are collected by subsystem index.

Different subsystem `PhiMap` callbacks may execute at the same time. Callbacks must not mutate shared state without synchronization. Use serial mode when callbacks intentionally share mutable state. The two-oscillator benchmark may be slower in parallel because its subsystem solves are cheaper than task dispatch; parallel execution benefits larger subsystem solves.

## Custom DRS stopping rules

`K_n` is the fixed maximum budget. An optional predicate can accept a completed inner iteration and stop before the next one:

```cpp
DrsOptions options;
options.K_n = 50;
options.stop_predicate = [](const DrsIterationView& it) {
  return it.residual < 1e-8;
};
```

The view exposes `k`, `u_current`, `u_next`, `b_hat`, `w`, and the residual as read-only values. Returning true accepts the current `u_next` and shadow. Exceptions propagate before any physical macro-step state is committed.

## Two-oscillator benchmark

The benchmark exercises the full generalized-pH construction:

```bash
./build/two_oscillators_benchmark
```

Subsystem A uses flow-input causality and its separable Duffing discrete gradient supplies an analytic next-state Jacobian. Subsystem B uses effort-input causality and a quadratic midpoint discrete gradient, which selects the affine direct-solve path. Each output directory includes `metadata.json` with the `generalized-ph` provenance and the full schema-v3 experiment configuration. The complete derivation, from the physical sketch through the matrix entries and the diagnostic CSV columns, is in [examples/two_oscillators_duffing_linear/TUTORIAL.md](examples/two_oscillators_duffing_linear/TUTORIAL.md).

## Outputs and reproduction

The benchmark writes `reference_bigK.csv`, per-K trajectory and diagnostic CSV files, and `summary.json`. `examples/two_oscillators_duffing_linear/plot_results.py` generates PNG, EPS, and outlined EPS. `--no-diagnostics` skips diagnostic computation; `--workers 1` forces the serial execution path.

`plot_results.py` requires Python 3 with `numpy` and `matplotlib`. EPS output uses `matplotlib`'s `text.usetex=True` mode, which shells out to a system LaTeX installation. Run `python examples/two_oscillators_duffing_linear/plot_results.py --help` for the figure-selection flags.

## Troubleshooting

Common failures, ordered by likelihood when first running the code.

**`PortHamiltonianModel J must be skew-symmetric`** (or `R must be symmetric`, `S must be symmetric`, `N must be skew-symmetric`). A matrix block has the wrong symmetry. Check $J_{ij} = -J_{ji}$, $R = R^\top$, $S = S^\top$, $N = -N^\top$. The tolerance is `structure_tolerance` (default $10^{-12}$, scaled by the matrix norm). Typos in floating-point entries usually fail by orders of magnitude beyond the tolerance.

**`PortHamiltonianModel block dissipation matrix [R P; P^T S] must be PSD`**. The combined dissipative block is indefinite. Either $R$ is too small to dominate a negative $P$ entry, or a $P$ entry has the wrong sign. Re-derive which damping share belongs to $R$ vs $P$ using the design rule in [TUTORIAL.md §4](examples/two_oscillators_duffing_linear/TUTORIAL.md) ("own-state share into $R$, neighbor-state share into $P/S$").

**`Dirac interconnection violates E F^T + F E^T = 0`** or **`Dirac interconnection [E F] must have full row rank`**. Hand-written $E$, $F$ matrices do not encode a lossless constraint. Prefer the factories (`equal_effort_opposite_flow`, `zero_junction`, `one_junction`, `ideal_transformer`, `ideal_gyrator`) over `from_effort_flow_constraints`; they construct valid constraints by construction.

**`Compiled interconnection resolvent is singular`**. The wave map $2I - U$ is not invertible, which means $U$ has an eigenvalue at 2. This should not happen for any valid lossless interconnection (orthogonal $U$ has eigenvalues on the unit circle). If it occurs, the input $E$, $F$ is on the boundary of the lossless cone; perturb and re-check.

**`Discrete-gradient Newton did not converge`**. The per-subsystem nonlinear solve stalled. Causes, in order of likelihood: (a) `dt` too large for the local stiffness, (b) poor initial guess far from a physically reasonable state, (c) wrong Jacobian (if you supplied `residual_jacobian`, verify it against a finite-difference check). Try halving `dt` first.

**`Discrete-gradient Newton line search failed`**. Newton produced a direction that did not reduce the residual even at half-step. Usually means the iterate is in a stiff or near-singular region. Same remedies as above; also try supplying an analytic discrete-gradient Jacobian (see [two_oscillator_model.cpp:72-82](examples/two_oscillators_duffing_linear/two_oscillator_model.cpp#L72-L82) for the pattern).

**`Discrete-gradient affine solve did not satisfy tolerance`**. You used `DiscreteGradient::quadratic_midpoint` but the residual is not affine. Either the Hamiltonian is not purely quadratic in `x` (check for nonlinearities), or the `Q` you passed is inconsistent with the model's `H`.

**FNE margin `min_fne_margin` goes negative in `summary.json`**. Condition 1 fails along the realized trajectory. The scattering impedance $\gamma$ is larger than the smallest incremental port impedance somewhere. Lower `gamma` and re-run; verify the margin stays nonnegative.

**`max_passivity_pos` or `max_augmented_pos` larger than $\sim 10^{-12}$**. Condition 2 or Theorem 1 is violated beyond roundoff. For a structure-preserving discrete gradient this almost always means the wrong integrator was used (e.g., called `PhiMap` directly with a non-discrete-gradient stepper). If using `make_discrete_gradient_phi`, check the Newton convergence log; iterates that barely converged can leak energy.

**DRS `inner_residual` does not decrease across $K_n$**. Either $\gamma$ is too large (the FNE margin is near zero and DRS is near a non-contractive region) or the subsystem is not FNE. Run the diagnostic path (`--diagK` larger than $K_n$) and inspect `fne_min_delta_*` over macro-steps.

**`LaTeX Error` or `latex not found` when running `plot_results.py`**. Install a TeX distribution (MacTeX on macOS, TeX Live on Linux) or set `matplotlib.rcParams['text.usetex'] = False` locally for PNG-only output.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wei2026early,
  title   = {Early-Terminable Energy-Safe Iterative Coupling for Parallel
             Simulation of Port-{H}amiltonian Systems},
  author  = {Wei, Qi and Tao, Jianfeng and Nie, Hongyu and Tan, Wangtao},
  year    = {2026},
  note    = {arXiv:2603.16424}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
