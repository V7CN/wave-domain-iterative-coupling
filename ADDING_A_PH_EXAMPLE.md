# Creating a New Example

An example has three components: passive subsystem steppers, a physical effort/flow interconnection, and numerical simulation settings. Algorithm 1 only sees the resulting `PhiMap` objects and the prepared wave interconnection.

## 1. Define each subsystem

Choose a state order and write it next to the model. Define the Hamiltonian, its gradient, the constant generalized-pH matrices, and the port causality:

For the paper's standard effort-input form $\dot x=(J-R)\nabla H+Ge$, $f=G^\top\nabla H$, use the compact constructor:

```cpp
PortHamiltonianModel model(
    state_dimension,
    port_dimension,
    hamiltonian,
    gradient,
    J, R, G);
```

For nonzero $P/S/N$ or flow-input causality, use the generalized constructor and state the causality explicitly:

```cpp
PortHamiltonianModel model(
    state_dimension,
    port_dimension,
    hamiltonian,
    gradient,
    J, R, G, P, S, N,
    PortCausality::EffortInputFlowOutput);
```

The subsystem convention is

$$
\dot x = (J - R)\, \nabla H(x) + (G - P)\, u, \qquad
y = (G + P)^\top \nabla H(x) + (S + N)\, u.
$$

Here the subsystem matrix $P$ belongs to its dissipative input/output structure. It is not the external coupling matrix called $P$ in the paper. The constructor checks dimensions, finiteness, skew/symmetric structure, and positive semidefiniteness of $\begin{pmatrix} R & P \\ P^\top & S \end{pmatrix}$.

## 2. Choose the discrete gradient

For a quadratic Hamiltonian

```cpp
Matrix Q = ...;
DiscreteGradient dg = DiscreteGradient::quadratic_midpoint(Q);
```

This selects the affine direct solve. For a general smooth Hamiltonian, the shortest correct definition is

```cpp
DiscreteGradient dg = DiscreteGradient::gonzalez(model);
```

For a separable nonlinear Hamiltonian, provide the exact discrete gradient and optionally its derivative with respect to the next state:

```cpp
DiscreteGradient dg(
    state_dimension,
    discrete_gradient,
    jacobian_with_respect_to_x_next);
```

The derivative selects analytic Newton. Without it, the common integrator uses its centered finite-difference Jacobian. Convert the model to the subsystem contract with

```cpp
PhiMap phi = make_discrete_gradient_phi(model, dg, integrator_options);
```

## 3. Define the physical interconnection

For the paper's two-subsystem relation

$$
e_A = e_B, \qquad f_A + f_B = 0,
$$

use

```cpp
auto interconnection =
    LinearDiracInterconnection::equal_effort_opposite_flow(port_dimension);
```

Other built-in physical relations are

```cpp
LinearDiracInterconnection::zero_junction(port_count, port_dimension);
LinearDiracInterconnection::one_junction(port_count, port_dimension);
LinearDiracInterconnection::ideal_transformer(ratio, port_dimension);
LinearDiracInterconnection::ideal_gyrator(gyration, port_dimension);
```

For a general constant linear lossless network, stack ports in the same order as the subsystem vector and define

$$
E\, e + F\, f = 0.
$$

Then construct

```cpp
auto interconnection =
    LinearDiracInterconnection::from_effort_flow_constraints(E, F);
```

`E` and `F` must be finite $m \times m$ matrices satisfying

$$
\operatorname{rank}(\begin{pmatrix} E & F \end{pmatrix}) = m, \qquad
E F^\top + F E^\top = 0.
$$

Invalid or non-lossless constraints are rejected; there is no implicit or least-squares fallback.

## 4. Configure and run Algorithm 1

Collect the subsystem steppers and initial states, choose the initial stacked outgoing wave, and set the numerical parameters:

```cpp
Wave b0 = Wave::Zero(total_port_dimension);

DrsOptions drs;
drs.K_n = 8;
drs.eps = 0.0;

PartitionedSimulator simulator(
    {{phi_A, x_A0}, {phi_B, x_B0}},
    b0,
    dt,
    gamma,
    interconnection,
    drs,
    ExecutionOptions{1});

SimulationLog log = simulator.run(number_of_steps);
```

The sum of subsystem port dimensions must equal the interconnection dimension and the size of `b0`. A `PhiMap` may be queried many times at a frozen state, so its callback must not mutate shared physical state.

To change the scalar scattering impedance between macro-steps:

```cpp
simulator.set_gamma_for_next_step(new_gamma);
```

The simulator preserves the represented physical effort/flow pair, re-scatters the committed wave, and prepares the new interconnection before the next step. Do not change `gamma` inside a macro-step.

### Choosing `dt`, `gamma`, and `K_n`

These three parameters govern accuracy, passivity margin, and wall-clock cost. They interact, so the guidance below treats them together.

**Macro-step `dt`.** Still picked by accuracy. The discrete-gradient stepper is structure-preserving (Condition 2 holds for any `dt`), but its local error is $O(\Delta t^2)$. Passivity guarantees that the simulation will not blow up; it does not replace accuracy analysis. Start with the largest `dt` consistent with your accuracy target and halve it if RMS error against a smaller-`dt` reference is too large. The two-oscillator benchmark uses $\Delta t = 0.01\,\mathrm{s}$ on a timescale of $T = 10\,\mathrm{s}$.

**Scattering impedance `gamma`.** Must satisfy $\gamma \le \lambda_{\min}(Z_d)$ where $Z_d$ is the incremental port impedance (paper Remark 1). For a linear system compute $Z_d$ once and pick `gamma` at or below its smallest eigenvalue. For a nonlinear system linearize along a representative trajectory and take the minimum. If $Z_d$ is unknown, scan `gamma` in $\{0.1, 0.4, 1.0, \sqrt{k\,m}, \dots\}$ and keep the value for which the FNE margin `min_fne_margin` (in `summary.json`) stays nonnegative along the run. The two-oscillator benchmark uses $\gamma = 0.4\,\mathrm{N\,s/m}$, well below $\sqrt{k_1 m_1} \approx 28\,\mathrm{N\,s/m}$.

**Inner budget `K_n`.** Controls how close each macro-step comes to the monolithic solve. $K_n = 0$ is the explicit wave interface (cheapest, least accurate). Empirically $K_n \in [5, 20]$ drives RMS state error to $\sim 10^{-3}$ for stiff problems; $K_n \ge 50$ reaches numerical roundoff against a long-`K_n` reference. Always verify monotonic RMS decay by sweeping a few values; non-monotonic decay indicates that `gamma` is too large or the subsystem is not FNE.

The benchmark writes the FNE, discrete-passivity, and augmented-storage residuals per macro-step into `diagnostics_KN.csv` and their per-`K_n` worst cases into `summary.json`. Use those columns to confirm the certificates hold along your trajectory before trusting any accuracy number.

## 5. Check a new example

At minimum, test:

- Hamiltonian and gradient dimensions and finite values;
- one frozen `PhiMap` step for several incident waves;
- discrete passivity residuals;
- physical interconnection constraints recovered from `(a,b)`;
- `K_n=0`, a finite `K_n`, and a long-reference run;
- serial/parallel equality when callbacks are thread-safe;
- the benchmark binary runs end-to-end and `summary.json` shows nonnegative FNE margins and passivity residuals at roundoff.

The two-oscillator model demonstrates nonlinear analytic Newton and the affine direct path. `generic_ph_mass_spring` is the minimal quadratic starting point.
