# Two-Oscillator Benchmark

This tutorial walks the benchmark used in Section V of the paper, from the physical sketch through the generalized port-Hamiltonian (pH) construction, the discrete-gradient stepper, the wave-domain coupling, and the diagnostic CSV columns.

Source files referenced below are in this directory:

- Physics and generalized-pH matrices: [two_oscillator_model.cpp](two_oscillator_model.cpp)
- Algorithm 1 wiring: [benchmark_runner.cpp](benchmark_runner.cpp)
- Offline diagnostics: [benchmark_diagnostics.cpp](benchmark_diagnostics.cpp)
- Configuration defaults: [benchmark_config.hpp](benchmark_config.hpp)

## 1. What this benchmark claims

Subsystem A is a Duffing oscillator (mass $m_1$, ground spring $k_1$, cubic hardening $k_{nl}$). Subsystem B is a linear oscillator (mass $m_2$, ground spring $k_2$). They are joined by a coupling spring $k_{12}$ and dashpot $c_{12}$. The benchmark simulates the coupled pair for $T = 10\,\mathrm{s}$ at $\Delta t = 0.01\,\mathrm{s}$ and scans the inner-iteration budget $K_n$ over $\{0, 3, 8, 20, 35, 50\}$.

For each $K_n$, three certificates are checked along the realized trajectory:

- Condition 1 (FNE of the frozen port map, Eq. 22).
- Condition 2 (discrete passivity of each subsystem integrator, Eq. 26).
- Theorem 1 (finite-iteration augmented storage, Eq. 31).

And one convergence claim:

- Theorem 2: as $K_n$ grows, the partitioned trajectory approaches the monolithic discrete-gradient reference.

Paper Figs. 4-6 plot the corresponding CSVs in `examples/two_oscillators_duffing_linear/out_cpp/`.

## 2. Physical system

Two unit-port mechanical subsystems sharing one coupling element.

```text
   k1, knl(q1^4)               k12, c12                 k2
  |--||-------+-------------+-/\/\/\-+-------------+------||---> x
              m1                                 m2
   ground                                                ground
```

Equations of motion (no external forcing):

$$
m_1 \ddot q_1 + c_1 \dot q_1 + k_1 q_1 + k_{nl} q_1^3 =  F_c,
$$

$$
m_2 \ddot q_2 + c_2 \dot q_2 + k_2 q_2            = -F_c,
$$

$$
F_c = k_{12} (q_2 - q_1) + c_{12} (\dot q_2 - \dot q_1).
$$

$F_c$ is the force transmitted through the coupling element, applied with opposite signs on the two masses. Default parameters (Table I in the paper) are encoded in [benchmark_config.hpp:23-31](benchmark_config.hpp#L23-L31) and in [two_oscillator_model.hpp:11-21](two_oscillator_model.hpp#L11-L21). The complete initialized state is $q_1(0)=0.4$, $v_1(0)=0$, $\delta(0)=q_2(0)-q_1(0)=-0.4$, $q_2(0)=0$, and $v_2(0)=0$; the committed outgoing wave starts from $b_A^0=b_B^0=0$.

## 3. Subsystem decomposition

The system is split at the coupling port. Each side becomes a single-port subsystem with state dimension 3 and 2.

| | A | B |
|---|---|---|
| Holds | mass $m_1$, ground spring $k_1$, cubic spring $k_{nl}$, and the coupling spring $k_{12}$ | mass $m_2$ and ground spring $k_2$ |
| State | $x_A = (q_1, v_1, \delta)$ with $\delta \equiv q_2 - q_1$ | $x_B = (q_2, v_2)$ |
| Hamiltonian | $H_A = \tfrac12 m_1 v_1^2 + \tfrac12 k_1 q_1^2 + \tfrac14 k_{nl} q_1^4 + \tfrac12 k_{12} \delta^2$ | $H_B = \tfrac12 m_2 v_2^2 + \tfrac12 k_2 q_2^2$ |
| Port exposes | the velocity of mass 2 (so that $\dot\delta = v_2 - v_1$ closes through the port) | the reaction force from the coupling |

Putting $\delta$ in subsystem A makes the coupling spring's potential internal to A; the only signal crossing the port is the kinematic variable $v_2$. The split is asymmetric on purpose: the harder subsystem (Duffing) is also the one that absorbs the coupling spring.

## 4. Subsystem A in generalized pH form

### Why the standard pH form is not enough

A first instinct is to plug subsystem A into the paper's Eq. (1) form
$\dot x = (J-R)\nabla H + Ge$, $f = G^\top \nabla H$ with $x_A = (q_1, v_1)$
and to handle the coupling through the port. That fails here because the
coupling dashpot force $c_{12}(v_2 - v_1)$ depends on $v_2$, which is not a
state of A. Two ways out:

1. Add $v_2$ to $x_A$. This breaks the partition: A would have to know B's
   state, defeating the point of partitioned simulation.
2. Let $v_2$ enter through the port as the input $u_A$. Then $u_A$ must
   appear in the $\dot v_1$ equation with the right coefficient, and the
   output $y_A$ must be the coupling force so that $u_A\, y_A$ equals the
   physical power delivered by mass 2.

Option 2 is what the generalized pH form supports. The extra matrices
$P$, $S$, $N$ are the price: they let the same input $u$ contribute to both
$\dot x$ and $y$, which is what the dashpot's $c_{12} v_2$ term
requires.

### The generalized pH convention

The library convention (see [README §Subsystems](../../README.md#subsystems-port-hamiltonian-models)) is

$$
\dot x = (J - R)\, \nabla H(x) + (G - P)\, u,
$$

$$
y = (G + P)^\top \nabla H(x) + (S + N)\, u, \qquad \text{supplied power} = u^\top y.
$$

### State and Hamiltonian

$$
x_A = (q_1, v_1, \delta), \qquad
\nabla H_A = \begin{pmatrix} k_1 q_1 + k_{nl} q_1^3 \\ m_1 v_1 \\ k_{12} \delta \end{pmatrix}.
$$

### Why J looks the way it does

Mechanical pH systems have a canonical skew-symmetric interconnection:

$$
J_A = \begin{pmatrix}
0 & 1/m_1 & 0 \\
-1/m_1 & 0 & 1/m_1 \\
0 & -1/m_1 & 0
\end{pmatrix}.
$$

The $(1,2)$ and $(2,1)$ entries give $\dot q_1 = v_1$ (the standard position-velocity pair). The $(2,3)$ and $(3,2)$ entries couple momentum to the relative displacement $\delta$, which is what makes the coupling spring pull on mass 1. See [two_oscillator_model.cpp:34-38](two_oscillator_model.cpp#L34-L38).

### Why R has a single nonzero entry

$$
R_A = \begin{pmatrix}
0 & 0 & 0 \\
0 & (c_1 + c_{12})/m_1^2 & 0 \\
0 & 0 & 0
\end{pmatrix}.
$$

Damping enters the momentum equation as $R\, \nabla H$, which contributes $-((c_1 + c_{12})/m_1^2) \cdot (m_1 v_1) = -(c_1 + c_{12})/m_1 \cdot v_1$ to $\dot v_1$. That is the combined ground damping $c_1 v_1$ plus the share of $c_{12}$ attributed to mass 1 in the split. The PSD check at construction accepts a single positive diagonal; see [port_hamiltonian_model.hpp:133-144](../../include/iterative_coupling/ph/port_hamiltonian_model.hpp#L133-L144).

### Why G is at row 3

$$
G_A = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.
$$

The port input enters the $\delta$-equation, so $\dot\delta = -v_1 + u_A$. Combined with $u_A = v_2$ (chosen below), this gives $\dot\delta = v_2 - v_1 = \frac{d}{dt}(q_2 - q_1)$, the correct relative velocity. See [two_oscillator_model.cpp:41-42](two_oscillator_model.cpp#L41-L42).

### Why P, S are nonzero and N is zero

The coupling dashpot exerts $c_{12}(v_2 - v_1)$ on mass 1. This force is split across two channels:

- The $R_{22} = c_{12}/m_1^2$ entry above already accounts for the $-c_{12} v_1$ share via the $R\, \nabla H$ term, contributing $-c_{12} v_1 / m_1$ to $\dot v_1$.
- The remaining $+c_{12} v_2$ share must enter through the port input $u_A = v_2$, so $u_A$ must appear in the $\dot v_1$ equation with coefficient $+c_{12}/m_1$. In Eigen's zero-based indexing, setting `P(1, 0) = -c12/m1` gives this coefficient through $(G - P)\,u$.

The $S = c_{12}$ block adds $c_{12} u_A$ to the output $y_A$, so that the supplied power

$$
\begin{aligned}
u_A\, y_A &= u_A \bigl( (G + P)^\top \nabla H + S\, u_A \bigr) \\
          &= u_A \bigl( -c_{12} v_1 + k_{12} \delta + c_{12} u_A \bigr) \\
          &= \bigl( k_{12} \delta + c_{12} (u_A - v_1) \bigr)\, u_A \\
          &= \bigl( k_{12} (q_2 - q_1) + c_{12} (v_2 - v_1) \bigr)\, v_2,
\end{aligned}
$$

matches the physical power delivered by mass 2 onto the coupling element, the $e^\top f$ identity (Eq. 5) in wave coordinates. See [two_oscillator_model.cpp:43-46](two_oscillator_model.cpp#L43-L46).

**Design rule for splitting a coupling dissipation.** The same pattern generalizes to any port-shared damper or, more broadly, to any force term that depends on a neighbor's state:

- The share driven by **this subsystem's own state** goes into $R$ (internal dissipation, no port involvement).
- The share driven by **the neighbor's state** enters through the port input $u$, which forces a matching entry in $P$ (so $u$ appears in $\dot x$ with the right coefficient) and in $S$ (so $u$ also appears in $y$ to close the supplied-power identity).
- $N$ stays zero unless you have a gyroscopic-style cross-coupling between distinct input components; the two-oscillator benchmark never needs it.

For a system with several coupling elements, repeat the split per element and sum the contributions into the same matrix blocks. The constructor's PSD check on $\begin{pmatrix} R & P \\ P^\top & S \end{pmatrix}$ will reject any combination that is not physically dissipative.

### Port causality

`PortCausality::FlowInputEffortOutput` declares $u_A = f_A$ (flow input) and $y_A = e_A$ (effort output). Here the flow is the velocity of mass 2 and the effort is the coupling force $F_c$. The discrete-gradient adapter uses this to assemble the scattering boundary condition; see [discrete_gradient_phi_detail.hpp:99-106](../../include/iterative_coupling/integrators/detail/discrete_gradient_phi_detail.hpp#L99-L106).

## 5. Subsystem B in generalized pH form

Subsystem B is simpler because $H_B$ is purely quadratic and there is no internal coupling spring to split.

$$
x_B = (q_2, v_2), \qquad \nabla H_B = \begin{pmatrix} k_2 q_2 \\ m_2 v_2 \end{pmatrix},
$$

$$
J_B = \begin{pmatrix} 0 & 1/m_2 \\ -1/m_2 & 0 \end{pmatrix}, \quad
R_B = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}, \quad
G_B = \begin{pmatrix} 0 \\ -1/m_2 \end{pmatrix}, \quad
P_B = S_B = N_B = 0.
$$

The port input enters the momentum equation with $G_{10} = -1/m_2$, so the coupling effort appears as $-u_B / m_2$ in $\dot v_2$. With `PortCausality::EffortInputFlowOutput`, $u_B = e_B$ is the coupling force and $y_B = f_B = -v_2$. The sign on $-v_2$ is what makes $f_A + f_B = v_2 - v_2 = 0$ hold at the port (Eq. 7). See [two_oscillator_model.cpp:87-106](two_oscillator_model.cpp#L87-L106).

## 6. Discrete gradient selection

Both subsystems use a discrete-gradient rule so that Condition 2 (Eq. 26) holds by construction. The library provides three entry points; the benchmark exercises two of them.

### Subsystem A: separable exact discrete gradient

$H_A$ is a sum of single-variable terms, so each term admits a closed-form exact discrete gradient:

| Term in $H_A$ | Exact discrete gradient (between $x_0$ and $x_1$) |
|---|---|
| $\tfrac12 k_1 q_1^2$ | $\tfrac12 k_1 (q_{1,0} + q_{1,1})$ |
| $\tfrac14 k_{nl} q_1^4$ | $\tfrac{k_{nl}}{4} (q_{1,0}^3 + q_{1,0}^2 q_{1,1} + q_{1,0} q_{1,1}^2 + q_{1,1}^3)$ |
| $\tfrac12 m_1 v_1^2$ | $\tfrac12 m_1 (v_{1,0} + v_{1,1})$ |
| $\tfrac12 k_{12} \delta^2$ | $\tfrac12 k_{12} (\delta_0 + \delta_1)$ |

Each row satisfies the defining exactness property $\overline{\nabla H}(x_0, x_1) \cdot (x_1 - x_0) = H(x_1) - H(x_0)$. Sum the four rows to obtain the discrete gradient used in [two_oscillator_model.cpp:58-71](two_oscillator_model.cpp#L58-L71).

The analytic $\partial / \partial x_1$ Jacobian of that discrete gradient is also closed-form ([two_oscillator_model.cpp:72-82](two_oscillator_model.cpp#L72-L82)). Supplying it switches the adapter to the analytic-Newton kernel, and because $n_A = 3$ and $m_A = 1$, the fixed-size `analytic_newton_step_3_1` fast path in [discrete_gradient_phi_detail.hpp:302](../../include/iterative_coupling/integrators/detail/discrete_gradient_phi_detail.hpp#L302) is selected. The frozen-step baseline test in [benchmark_tests.cpp:143-145](../../tests/benchmark_tests.cpp#L143-L145) locks the number of Newton iterations, so an accidental regression to the finite-difference Jacobian would be caught.

### Subsystem B: quadratic midpoint

$H_B$ is purely quadratic in $x_B$, so the midpoint discrete gradient $\tfrac12 Q (x_0 + x_1)$ is exact. The library provides a dedicated constructor that also flags the residual as affine, selecting the direct-solve path (no Newton iteration, single linear solve per step). See [discrete_gradient.hpp:84-114](../../include/iterative_coupling/integrators/discrete_gradient.hpp#L84-L114). For a quadratic Hamiltonian this rule coincides with the implicit midpoint (trapezoidal) update, which is what the paper means when it says the linear subsystem reduces to midpoint. Code at [two_oscillator_model.cpp:107-111](two_oscillator_model.cpp#L107-L111).

## 7. The coupling port

The two-port lossless constraint (Eq. 7) is

$$
e_A = e_B, \qquad f_A + f_B = 0.
$$

The library accepts this directly in physical coordinates:

```cpp
auto interconnection =
    LinearDiracInterconnection::equal_effort_opposite_flow(1);
```

`prepare(gamma)` compiles the constraint to the orthogonal wave map $a = P b$ (Eq. 9). For the `equal_effort_opposite_flow` factory with `port_dimension = 1`, $P$ is the $2 \times 2$ swap matrix. The compiled object also factorizes $2I - P$ once, so the per-step DRS resolvent $J_L = (2I - P)^{-1}$ (Eq. 16) is a cached QR solve, not a fresh factorization. The `test_linear_dirac_interconnections` test in [core_tests.cpp:87-214](../../tests/core_tests.cpp#L87-L214) checks the wave map numerically against the swap matrix and against the recovered physical $(e, f)$ pair.

The C++ name `wave_map_` (and `U` in some comments) deliberately differs from the paper's $P$ to avoid clashing with the $P$ matrix of the generalized-pH subsystem. See the note in [prepared_wave_interconnection.hpp:17-20](../../include/iterative_coupling/coupling/prepared_wave_interconnection.hpp#L17-L20).

## 8. Wiring up Algorithm 1

The minimal call sequence is four lines once the model and interconnection are in hand:

```cpp
const auto model = two_oscillator_example::make_two_oscillator_model(
    config.physics, config.initial, config.model_solver);
const auto interconnection =
    LinearDiracInterconnection::equal_effort_opposite_flow(1);

DrsOptions drs_options;
drs_options.K_n = K;   // inner-iteration budget

PartitionedSimulator simulator(
    {{model.phi_A, model.x_A0}, {model.phi_B, model.x_B0}},
    model.b0, config.dt, config.gamma,
    interconnection, drs_options, config.execution);

SimulationLog log = simulator.run(config.steps);
```

Mapping to Algorithm 1 in the paper:

- Constructing `PartitionedSimulator` stores the committed states $x_A^0$, $x_B^0$, the initial outgoing wave $b^0$, and the prepared interconnection.
- Each call to `PartitionedSimulator::step(n)` executes one outer macro-step of Algorithm 1: freeze state (line 3), initialize $u^{n,0} = P b^n$ (line 6) and $\hat b^{n,\star} = b^n$ (line 7), run the inner loop (lines 9-13), check the optional residual tolerance (lines 14-16), and finally advance both subsystems in parallel on $\hat a^n = P\, \hat b^{n,\star}$ (lines 19-21).
- Setting $K_n = 0$ skips the inner loop and reduces the macro-step to an explicit wave interface, the $K_n = 0$ case of Eq. (19).

The production wiring (with per-step diagnostics via `FrozenStepObserver`) is at [benchmark_runner.cpp:43-65](benchmark_runner.cpp#L43-L65); the short snippet above is what the algorithm itself needs.

## 9. Reading the diagnostics

When `--no-diagnostics` is not set, the benchmark writes one `diagnostics_KN.csv` per $K_n$. Each row is one macro-step, with columns defined by [AuditRow](benchmark_diagnostics.hpp#L13-L23) and written by [benchmark_io.cpp:61-72](benchmark_io.cpp#L61-L72). By default, each subsystem's FNE check uses 32 normally distributed test pairs with scale 1 and tolerance $10^{-12}$; the deterministic seeds at macro-step $n$ are $1000+n$ for A and $2000+n$ for B. The columns map to the paper as follows.

| Column | Paper object | Offline-only? |
|---|---|---|
| `fne_min_delta_A`, `fne_min_delta_B` | Condition 1 margin, Eq. 22 rearranged: $\langle S(\alpha) - S(\beta),\, \alpha - \beta \rangle - \| S(\alpha) - S(\beta) \|^2 \ge 0$ | yes, sampled random-pair test |
| `fne_viol_A`, `fne_viol_B` | count of test pairs violating the above within `fne_tolerance = 1e-12` | yes |
| `rA`, `rB` | Condition 2 residual $r_i^n = \Delta H_i - \tfrac12 (\| a_i^n \|^2 - \| b_i^{n+1} \|^2)$, expected $\le 0$ | online, also stored in `StepLog::energy` |
| `augmented_residual` | Theorem 1, Eq. 31: LHS $-$ RHS with $u^{n,\dagger}$ approximated by a long DRS (`reference_iterations = 80`) | yes, the $u^{n,\dagger}$ approximation is diagnostic-only |
| `inner_residual` | $\| u^{n,k+1} - u^{n,k} \|$, the per-step Fejér length, Eq. 27 | online |
| `inner_iterations` | effective inner count, may be smaller than $K_n$ if `eps` triggered early exit | online |

The diagnostics run inside the `FrozenStepObserver` hook ([benchmark_runner.cpp:45-54](benchmark_runner.cpp#L45-L54)), which the simulator calls after freezing $S^n$ and before the online DRS solve. The frozen map passed to the hook is the one used by Algorithm 1, so diagnostics and online step share the same operator.

Three summary numbers in `summary.json` aggregate each $K_n$ run:

- `max_passivity_pos`: $\max_n \max_i (r_i^n)_+$. Paper claim: at numerical roundoff (around $10^{-14}$).
- `max_augmented_pos`: $\max_n (\text{augmented\_residual}_n)_+$. Same roundoff claim.
- `min_fne_margin`: $\min_n \min_i \Delta_i^n$. Paper claim: nonnegative on the realized trajectory.

The paper calls the convergence quantity the RMS state error. In this benchmark implementation, the reported error vector is formed from the two plotted position components, $[q_1-q_{1,\mathrm{mono}},\,q_2-q_{2,\mathrm{mono}}]^\top$; `rms_error` is its root mean square over the stored time grid. The monolithic comparison is stored as `reference_bigK.csv` and is generated by the configured reference solve (`refK=200` by default). The filename and `refK` setting are internal; figures and discussion use the paper's monolithic-reference terminology.

## 10. Reproducing the paper figures

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/two_oscillators_benchmark
pip install numpy matplotlib
python examples/two_oscillators_duffing_linear/plot_results.py examples/two_oscillators_duffing_linear/out_cpp
```

The CSV-to-figure mapping is:

| Paper figure | CSVs consumed | Claim |
|---|---|---|
| Fig. 4 | `reference_bigK.csv`, `trajectory_K*.csv` | $q_1(t)$ and $q_2(t)$ collapse onto the reference as $K_n$ grows |
| Fig. 5(a) | `diagnostics_K*.csv` (`fne_min_delta_*`, `fne_viol_*`) | Condition 1 margins are nonnegative along the trajectory |
| Fig. 5(b) | `diagnostics_K*.csv` (`rA`, `rB`, `augmented_residual`) | Condition 2 and Theorem 1 positive parts are at roundoff |
| Fig. 6(a) | `trajectory_K*.csv` minus the monolithic reference | RMS state error versus time decreases as the inner-iteration budget increases; the plotted implementation quantity uses $q_1,q_2$ |
| Fig. 6(b) | `summary.json` `rms_error` | RMS state error decays monotonically in $K_n$, supporting Theorem 2 |

The defaults in [benchmark_config.hpp](benchmark_config.hpp) match Table I. Override `--steps`, `--refK`, `--diagK`, `--fne-pairs`, `--workers`, and `--no-diagnostics` from the command line; see [benchmark_config.cpp:28-64](benchmark_config.cpp#L28-L64).
