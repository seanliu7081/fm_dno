# Research Design: Orbit-Aligned Source Coupling and Orbit-Structured Noise Optimization for Flow-Based Robot Policies

**Status.** Design + reference implementation + synthetic validation. No LIBERO runs yet.
**Base repo.** `seanliu7081/past2next_clean`, targeting `oat.policy.flow_policy.FlowPolicy`
(`train_flowpolicy.yaml`). Past-action / enriched-past designs are out of scope.
**Symmetry.** SO(2) about gravity.
**DNO role.** Test-time action steering of a frozen policy.

---

## 0. One-paragraph summary

The double-ring study established that on an SO(2)-symmetric target, *how you pair source
with target* matters more than *what the source marginal is*, and that an orbit-aligned
pairing buys a vanilla flow-matching MLP both better samples and stronger relaxed
equivariance. This document ports that to an action-chunk policy. The LIBERO 7-D delta-EEF
action decomposes under a rotation about gravity into two frequency-1 blocks and three
invariants — the chunk analogue of `[s, v_x, v_y]` — so the coupling transfers almost
verbatim. Three things do **not** transfer, and they are what the design is mostly about:
the repo's per-dimension action normalizer destroys the group structure before the network
ever sees it; the target heading distribution of robot demonstrations is *not* uniform,
which turns "rotate the source onto the target's orbit position" from a marginal-preserving
operation into a train/inference prior shift; and a fixed RGB camera means the textbook
equivariance property is not merely hard to measure but false by construction, so the claim
has to be restated in terms of a property that is true and testable. Working through the
second of those yields the mechanism the design is built on: an orbit-aligned coupling makes
the source's global phase *control* the generated chunk's heading, which turns the expensive
inner loop of diffusion noise optimization into a one-dimensional exhaustive search that fits
inside a control period. Synthetic validation confirms the mechanism and adds one thing the
unconditional toy problem could not have shown — that same control authority is *delegation*:
the policy stops inferring the heading from the observation, so open-loop conditional accuracy
drops by roughly 70% and is then more than recovered by the orbit search (3.8× more task-loss
reduction per forward pass than the same search on an iid-coupled policy, ending more
conditionally accurate than the iid policy). The coupling and the noise optimization are
therefore one contribution, not two that compose.

---

## 1. What transfers, and what does not

| Toy component | Robot analogue | Transfers? |
|---|---|---|
| `x = [s, v_x, v_y]`, SO(2) | `a = [dx,dy,dz,wx,wy,wz,g]`, SO(2) about z | Yes — two vector blocks instead of one |
| Single sample | Chunk `x ∈ R^{H×7}`, one shared angle | Yes — the group acts on the chunk |
| Target angles uniform | Demo headings strongly non-uniform | **No** — §5 |
| No normalizer | Per-dim min-max to `[-1,1]` | **No** — §3 |
| Unconditional | Conditioned on observation tokens | Partly — §6.3 |
| Equivariance of `f_θ(gx,t)` | Camera does not transform under `g` | **No** — §4 |
| Method D radial OT | Chunk-magnitude OT | Yes, subsumed by the cost choice — §6.1 |

---

## 2. The target: LIBERO action chunks under SO(2)

The `libero10` task config gives `action.shape = [7]`; robosuite's `OSC_POSE` controller
reads that as
\[
a=[\Delta p_x,\Delta p_y,\Delta p_z,\;\omega_x,\omega_y,\omega_z,\;g].
\]
Under a rotation \(R_\phi\) about the world \(z\) axis, \(\Delta p\) transforms as a vector
and \(\omega\) (axis-angle) as a pseudovector, which for a proper rotation is the same
thing. Decomposing into irreps of SO(2):

\[
\underbrace{(\Delta p_x,\Delta p_y)}_{\text{freq }1},\quad
\underbrace{(\omega_x,\omega_y)}_{\text{freq }1},\quad
\underbrace{\Delta p_z,\;\omega_z,\;g}_{\text{freq }0}.
\]

A chunk carries the action
\[
(\rho(\phi)x)_h=[R_\phi(\Delta p)_h^{xy},\,\Delta p_{z,h},\,R_\phi\omega_h^{xy},\,\omega_{z,h},\,g_h],
\]
with a **single \(\phi\) shared across the horizon** — the same convention the repo's
existing `SO3ActionChunkAug` already uses (one random rotation per chunk).

Everything is easier in a complex view: pack each 2-D block at each timestep into one
complex number, \(m = H\cdot 2\) of them, and \(\rho(\phi)\) becomes multiplication by
\(e^{i\phi}\).

> **Prerequisite to check before anything else.** This assumes world/base-frame deltas.
> If your `controller_configs` deliver end-effector-frame deltas, a world rotation acts
> *trivially* on the action and the entire symmetry statement changes. Read the config.

**Second prerequisite.** Report the per-block share of chunk energy after normalization
(`so2_chunk.vector_energy_fraction`). If \((\omega_x,\omega_y)\) turns out to carry a few
percent, the coupling is effectively translation-only; say so and run
`SO2ChunkSpec.translation_only()` as the honest version rather than implying both blocks
participate.

---

## 3. Prerequisite: the normalizer breaks the group action

`LinearNormalizer.fit(mode='limits')` — the repo default, inherited from Diffusion Policy —
fits **per dimension**:
\[
N(a)_d=s_d a_d+o_d,\qquad s_d=\tfrac{2}{\max_d-\min_d}.
\]
Then \(s_0\neq s_1\) and \(o\neq0\), so
\[
N(\rho(\phi)a)\neq\rho(\phi)N(a).
\]
The network sees a sheared, translated copy of the action space in which a rotation is not
a rotation. Every geometric object downstream — the coupling, the equivariance metrics, the
orbit stage of DNO — lives in that space.

The fix constrains the affine map to commute with \(\rho(\phi)\): one shared scale per
vector block and **exactly zero offset** there; unconstrained affine on the invariants.

Measured on synthetic chunks with LIBERO's irrep layout
(`scripts/validate_so2_coupling.py`, test 3):

| normalizer | \(\|N(\rho a)-\rho N(a)\|/\|\rho N(a)\|\) | scale(dx) | scale(dy) |
|---|---:|---:|---:|
| per-dim min-max (repo default) | 2.38e-01 | 1.0129 | 1.2929 |
| SO(2) block, `rms` | 3.48e-08 | 2.1405 | 2.1405 |
| SO(2) block, `quantile` | 2.64e-08 | 1.0387 | 1.0387 |

A 24% relative equivariance error from the normalizer alone dwarfs the ~10–27% coupling
effects the toy study was chasing. This is a required change, and it needs its own control
row (P0 vs P1 in §7) so that the normalizer change is never confounded with the coupling
change. I could not find prior work analysing this interaction; treat it as a small
contribution and a large hazard.

---

## 4. What "equivariance" can and cannot mean here

The textbook property is
\[
F_\theta(\rho_Z(g)z\mid\rho_C(g)o)=\rho_X(g)F_\theta(z\mid o).\tag{1}
\]
With LIBERO's fixed `agentview` camera, (1) is not merely unmeasurable — it is false by
construction. A global rotation of the world rotates the camera *and* the objects together,
so the rendered image is **invariant**, while the action must be **equivariant**. Hu et al.
(*3D Equivariant Visuomotor Policy Learning via Spherical Projection*, NeurIPS 2025
Spotlight, arXiv:2505.16969) state this explicitly; it is why the equivariant-manipulation
line (EquiBot, Diffusion-EDF, RiEMann) almost universally uses point clouds or segmented
depth, and why Equivariant Diffusion Policy leans on voxel/depth-derived inputs.

So the design measures three *different* things and never calls the first by the name of
the third (`oat/symmetry/metrics.py`):

1. **Source-orbit (noise-space) equivariance.** Hold \(o\) fixed, rotate only the source:
 \[
 E_{\text{src}}=\mathbb E\frac{\|F_\theta(\rho(\phi)z\mid o)-\rho(\phi)F_\theta(z\mid o)\|}{\|\rho(\phi)F_\theta(z\mid o)\|}.
 \]
 Measurable in the real RGB setting, and exactly the property orbit-DNO consumes.
2. **Orbit steering gain.** The same probe read as a control gain: radians of chunk-heading
 rotation per radian of source rotation. \(G=1\) means the noise phase steers the chunk;
 \(G=0\) means the network has learned to ignore it.
3. **Policy equivariance**, i.e. (1). Only available in a state-only or point-cloud
 variant. Provided so the state-obs ablation can report the honest number.

**Consequence for the claim.** In the RGB setting the paper cannot say "the coupling makes
a vanilla policy relaxed-equivariant" in sense (3). It can say: the coupling makes the
policy's *source space* group-structured (senses 1–2), and that this is what makes
test-time steering cheap. Sense (3) belongs to the state-obs ablation. Getting this
distinction wrong is the fastest way to lose a reviewer.

---

## 5. The obstruction the toy problem hides

### 5.1 Statement

In the double ring, target angles are uniform, so setting \(\theta_0:=\theta_1\) leaves the
source marginal exactly \(\mathcal N(0,I)\): the map replaces \(z\)'s orbit coordinate with
an independent uniform draw and touches nothing else.

Robot demonstrations are not uniform. Planar chunk headings are task- and scene-dependent
and often multimodal — the arm goes where the objects are. Then \(\theta_0\) inherits
\(p(\theta_1)\), while inference still draws \(z\sim\mathcal N(0,I)\). That is a
train/inference prior mismatch, and it is the single most likely way a naive port fails.

The obstruction is not an implementation defect. It is a lower bound:

> **If the source's angular law must remain uniform and the target's is not, the expected
> angular transport of any coupling is bounded below by the circular Wasserstein distance
> \(W_1(\mathrm{Unif},p_\theta)>0\).**

Optimal circular OT attains that bound; nothing marginal-preserving beats it. In the toy
problem \(p_\theta\) is uniform, the bound is 0, and the tension is invisible.

### 5.2 Measured (synthetic chunks, non-uniform headings, `W_1(\mathrm{Unif},p_\theta)=0.679`)

| coupling | path len ↓ | angle gap ↓ | source heading \(R_1\) (0 = undeformed) |
|---|---:|---:|---:|
| A iid | 8.221 | 1.598 | 0.054 |
| P1 perm, `group_ot` cost | 8.199 | 1.580 | 0.046 |
| **P2 perm, `angle` cost** | 8.139 | **0.698** | **0.059** |
| P3 perm, `euclidean` cost | 7.926 | 1.023 | 0.059 |
| C1 rot, `kabsch` align | 7.898 | 0.904 | 0.259 |
| C2 rot, `heading` align | 8.016 | **0.000** | 0.503 |
| C4 rot, `heading`, \(\kappa=2\) | 8.120 | 0.662 | 0.355 |
| F perm+rot, `group_ot` | **7.763** | 0.777 | 0.271 |

Three things to read off:

* **P2 sits on the theoretical bound** (0.698 vs 0.679). The marginal-exact optimum is
 circular OT on the heading, and the solver attains it.
* **C2's source \(R_1\) equals the target's \(R_1\)** (0.503 vs 0.502) — exactly as the
 orbit-reparameterization argument predicts for `align='heading'`. The deformation is not
 vague; it is characterized.
* **P1 is a no-op.** Matching on a rotation-invariant cost and then not rotating buys
 nothing: the cost deliberately ignores the coordinate you refused to change. `group_ot`
 is only meaningful in `perm_rot`.

### 5.3 Resolution: stop insisting the prior be isotropic

Flow matching requires the source you *sample from* to equal the source you *trained on* —
not that either be \(\mathcal N(0,I)\). So: train with `mode='rot', align='heading'`, then
sample from the induced law.

With `align='heading'` that law is exactly "\(\mathcal N(0,I)\) with the heading redrawn
from \(p_\theta\)": decompose \(z=(\theta,w)\) into heading and orbit-invariant remainder,
which are independent with \(\theta\sim U(0,2\pi)\) for an isotropic Gaussian; the coupling
replaces \(\theta\) and nothing else. `MatchedHeadingPrior` fits a circular histogram of the
sources actually used in training and re-imposes it at inference. No mismatch, full angular
transport benefit retained. (With `align='kabsch'` the replaced coordinate is
target-dependent, so the induced law is only approximately of that form — visible as the
residual coordinate-mean shift in the audit. That is the price of the transport-optimal
alignment, which is why both are implemented.)

This changes the toy study's Round-2 knob. The sweep is no longer "how strong a coupling",
it is a genuine two-axis Pareto plot: **transport reduction** against **prior deformation**
\(W_1(\text{source heading},\mathrm{Unif})\) — with the matched prior collapsing the second
axis to zero at the cost of a non-isotropic inference prior.

---

## 6. Coupling design

### 6.1 The design space

Two independent binary choices, plus a cost:

|  | do **not** apply \(\phi^\*\) | **apply** \(\phi^\*\) |
|---|---|---|
| identity pairing | `iid` (baseline A) | `rot` (toy Method C) |
| batch assignment | `perm` (marginal-exact) | `perm_rot` (Klein-style) |

\[
\text{cost: }\;
\texttt{group\_ot}\;\max\textstyle\sum_{ij}|S_{ij}|,\qquad
\texttt{euclidean}\;\max\textstyle\sum_{ij}\mathrm{Re}(S_{ij}),\qquad
\texttt{angle}\;\min\textstyle\sum d_\circ(\theta_i,\theta_j),
\]
where \(S_{ij}=\sum_k\overline{z_{i,k}}\,x_{j,k}\) is the complex cross-correlation of the
vector parts.

Two closed forms make this cheap. First,
\[
\min_\phi\|\rho(\phi)z-x\|^2=\|z\|^2+\|x\|^2-2|S|,\qquad \phi^\*=\arg S,
\]
the 2-D analogue of the Kabsch step. Second, \(\|z_i\|^2\) is constant along a row and
\(\|x_j\|^2\) along a column, so **neither affects the optimal assignment**: the group-aware
assignment depends on \(|S_{ij}|\) alone. That gives a one-line characterization of what
"symmetry-aware" means here:

\[
\boxed{\ \text{symmetry-blind uses }\mathrm{Re}(S_{ij});\quad\text{symmetry-aware uses }|S_{ij}|.\ }
\]

Chunk-magnitude matching (the analogue of the toy's radial OT, Method D) is not a separate
knob — it is what `euclidean` and `group_ot` already do through \(|S|\); `scalar_weight`
extends it to the invariant channels.

Complexity: \(O(B^2m)\) for the cost matrix plus a Hungarian solve, which at \(B=32\)–256 is
sub-millisecond on CPU. `cost='angle'` additionally admits an \(O(B\log B+B^2)\) pure-torch
circular-OT solver (sort both heading sets, search the \(B\) cyclic shifts — optimal plans on
the circle are cyclically monotone), so no scipy dependency is required.

### 6.2 Softening

`kappa` adds a von Mises dither to the applied rotation (\(\infty\) = exact alignment, the
toy's Round-1 setting); `coupling_prob` leaves a fraction of the batch iid; and
`min_heading_confidence` skips chunks whose planar deltas cancel over the horizon (a hold or
a reversal, where the heading carries no information). LIBERO demos contain many
near-stationary chunks, so the last one is not cosmetic.

### 6.3 The conditional-minibatch caveat

The assignment runs within a batch whose elements have *different* observations, so a source
can be handed to a target from another condition. Averaged over batches this is still a valid
coupling with the right marginals (Pooladian et al., *Multisample Flow Matching*, ICML 2023),
but the per-condition coupling is biased, and the bias grows with batch size at the same time
as the coupling strengthens. **Batch size is a coupling hyper-parameter here, not a free
training detail** — hold it fixed across the comparison and report it. This is also the most
likely explanation if a marginal-exact OT coupling improves transport metrics and *hurts*
sample quality (§8 shows exactly that pattern in the synthetic study).

`rot` has no such bias: it is per-sample and condition-respecting. So the two variants trade
different defects against each other, which is a reason to run both rather than pick one.

---

## 7. Why flow matching and not diffusion

Flow matching is defined for an arbitrary coupling \(\pi(z_0,x_1)\) with correct marginals;
the induced marginal path still connects \(p_0\) to \(p_1\). That licence is what makes
`rot` legitimate.

DDPM does not inherit it. Its forward process requires the noise to be conditionally
standard Gaussian *given the data*:
\[
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad
\varepsilon\sim\mathcal N(0,I)\ \perp\ x_0 .
\]
Rotating \(\varepsilon\) so its phase matches \(x_0\) makes \(p(\varepsilon\mid x_0)\)
non-Gaussian, changing the forward process the score is being fitted to while sampling still
starts from \(\mathcal N(0,I)\). So `rot` and `perm_rot` are **not valid** for the repo's
`DiffusionTransformerPolicy` / `DiffusionUnetPolicy`.

What is defensible for diffusion:

* `perm` only. A within-minibatch permutation preserves the batch's noise marginal exactly;
 only the per-sample conditional changes. This is precisely the construction and the
 argument of **Immiscible Diffusion** (Li et al., NeurIPS 2024, arXiv:2406.12303) — inherit
 their caveat (the paper itself notes \(p(x_t\mid x_0)\neq\mathcal N(0,I)\)) rather than
 re-deriving it as a theorem.
* Or reformulate as a bridge (DDBM, arXiv:2309.16948; I²SB) — the rigorous route, not taken
 here.
* Or use the flow policy, whose probability-flow ODE is the same object without the trouble.

`oat/policy/flow_policy_orbit.py` carries this as an executable warning: the diffusion class
raises rather than silently doing the unsound thing.

---

## 8. Orbit-structured DNO

### 8.1 The mechanism

Plain DNO treats the source as an unstructured vector in \(\mathbb R^{H\times A}\) and
searches it with hundreds of gradient steps through the full sampler. That is why the
literature has DNO doing offline motion editing (Karunratanakul et al., CVPR 2024), controlled
generation (D-Flow, ICML 2024, arXiv:2402.14017) and offline environment generation (ADTG =
*Adaptive Diffusion Terrain Generator*, CoRL 2024, arXiv:2410.10766) — and essentially never
closed-loop manipulation. DSRL (Wagenmaker et al., CoRL 2025, arXiv:2506.15799) gets to
real time only by amortizing the search into a learned actor.

The orbit-aligned coupling changes what the source space *is*. Because training paired each
source with a target on the same orbit, the source's global phase becomes the generated
chunk's heading. That hands DNO a **one-dimensional, compact, periodic** coordinate that
captures the single most consequential thing a manipulation policy gets wrong — which way to
go — and it is a coordinate you can search *exhaustively* instead of descending.

\[
\textbf{Stage 1 (orbit search)}\quad \phi^\*=\arg\min_{\phi\in\{2\pi k/K\}}\mathcal L\big(F_\theta(\rho(\phi)z\mid o)\big)
\]
— one batched forward pass over \(K\) rotations, no gradients, no local minima.

\[
\textbf{Stage 2 (residual descent)}\quad
z^\star=\arg\min_z \mathcal L(F_\theta(z\mid o))+\lambda_{\mathrm{typ}}R_{\mathrm{irrep}}(z)+\lambda_{\mathrm{tr}}\tfrac{\|z-z_{\mathrm{init}}\|^2}{D}
\]
— a handful of `IrrepAdam` steps from the stage-1 point.

### 8.2 The clean experiment

The 2×2: `{iid, orbit-coupled}` policy × `{stage 1 on, off}`. Note carefully what the
prediction is *not*: with steering gain 0, rotating an iid policy's source yields \(K\)
unrelated samples, so stage 1 degenerates into best-of-\(K\) resampling — which still helps.
The claim is the sharper one that **at identical compute the coupling converts a blind
best-of-\(K\) draw into a structured one-dimensional search**. Measured in the synthetic
study (§9.2): 3.8× more task-loss reduction per forward pass, and the coupled policy ends up
*more* conditionally accurate than the iid one despite starting *less* so.

### 8.3 The contract between the two halves

§9.1 shows that a steering gain of 1 has a cost: the policy stops inferring the chunk heading
from the observation and reads it off the noise phase instead, so open-loop conditional
accuracy drops. That is not a defect to regularize away — it is the contract. **A policy that
delegates the heading to the noise must be given the heading**, and stage 1 is how you give it.
Consequences for how this is presented and evaluated:

* The coupling and the DNO stage are **one contribution**, not two that compose. Reporting
 orbit-coupled open-loop numbers without the orbit search is reporting the ablation.
* The `kappa` sweep is the dial on how much heading authority to delegate: \(\kappa=\infty\)
 delegates fully (gain ≈ 1), \(\kappa=2\) delegates none (gain ≈ 0, and open-loop quality
 returns to baseline). Sweeping it traces the delegation frontier directly.
* If your deployment has no usable test-time objective, do not use a rotation-applying
 coupling. Use `perm` with `cost='angle'` or `'euclidean'`: it buys the transport and
 few-step benefits with far less delegation.

### 8.4 Group-compatible optimization

The DNO review's §5.3 is right that coordinate-wise Adam breaks rotation equivariance, and
the reason is the preconditioner: Adam's step is \(\eta\, m/(\sqrt v+\epsilon)\) with
diagonal \(v\), and a diagonal matrix commutes with an in-block rotation only when it is a
multiple of the identity there. Sharing \(v\) across the two coordinates of every frequency-1
block fixes it in one line; the first moment needs no treatment because it is linear in the
gradients and rotates covariantly. Measured (test 4), after 25 steps from \(z\) and from
\(\rho(0.7)z\) on an invariant objective:

| optimizer | \(\|z_k(\rho z_0)-\rho z_k(z_0)\|/\|z_k\|\) |
|---|---:|
| `IrrepAdam` | 1.61e-07 |
| plain Adam | 4.79e-01 |

The typical-set regularizer follows the review's §8.1: \(R(z)=\|z\|^2\) is wrong because a
\(d\)-dimensional Gaussian keeps its mass at \(\|z\|\approx\sqrt d\), so the penalty is the
negative log density of the norm, \(-(d-1)\log r+r^2/2\), applied **per irrep group** rather
than as one global norm — otherwise the optimizer trades the scale of the vector part
against the scalar part.

Warm start is the review's \(\mathrm{Shift}(z^\star)+\varepsilon\): after executing
\(n\) of \(H\) steps, the next chunk's first \(H-n\) steps cover the same interval as the old
chunk's tail, so sliding the solved noise along the horizon is the noise-space analogue of an
MPC warm start.

### 8.5 Budget

At 20 Hz with `n_action_steps=8`, a chunk is due every 0.4 s. For the repo's 4-layer /
256-dim transformer at `num_inference_steps=10`, a batch-1 forward is on the order of 5–10 ms
and a backward through the 10-step chain roughly 3×. So one stage-1 grid (\(K=32\) ≈ one
batch-32 forward) plus 6–10 gradient steps fits inside the period. Measure it —
`OrbitDNO.last_info['wall_time']` exists for that.

### 8.6 Task objectives

* **Tier 0** (no learning, no privileged state): chunk smoothness, seam continuity with the
 last executed action, per-step delta limits, workspace box, table clearance. Run this first
 — it isolates *steerability* from *whether the objective is a good proxy for success*.
* **Tier 1** (privileged, analysis only): obstacle SDFs from simulator object poses. Report
 as an oracle, never as a deployable result.
* **Tier 2** (learned): a success classifier or Q function on (obs, chunk) — the DSRL
 comparison, and the point where reward hacking dominates.

Note that most real task objectives **break** the symmetry (an axis-aligned workspace box has
only \(C_4\) symmetry; a table plane is invariant only if the scene rotates too). That is
fine and expected — it is why the *policy* is asked for relaxed rather than exact
equivariance — but it means the DNO stage itself is not equivariant and should not be
described as such.

---

## 9. Synthetic validation (run, not hypothesized)

`scripts/validate_so2_coupling.py` builds a chunk distribution with LIBERO's exact irrep
layout: four motion primitives (two speeds × two curvatures) rotated to a heading drawn from a
bimodal von Mises, with a conditioning vector that reveals the heading only noisily — the
realistic situation, since an agentview image *does* tell you roughly where the object is, it
just does not transform under a world rotation. Tests 1–4 are reported above (§3, §5.2, §8.3).
Test 5 trains a small conditional flow-matching model per coupling.

3 seeds × 2500 updates, SO(2) block normalizer throughout; the inference prior is
\(\mathcal N(0,I)\) except for the `+prior` rows.

| coupling | SWD ↓ | headMAE ↓ | 1-step gap ↓ | 4-step gap ↓ | straightness ↓ | src-orbit eq ↓ | **steer gain** |
|---|---:|---:|---:|---:|---:|---:|---:|
| A iid | **0.054** | **0.677** | 0.370 | 0.118 | 3.737 | 1.220 | −0.00 |
| P1 perm `group_ot` | **0.053** | 0.705 | 0.383 | 0.120 | 4.010 | 1.181 | −0.01 |
| P2 perm `angle` | 0.088 | 0.953 | 0.349 | 0.108 | 3.372 | 0.986 | −0.01 |
| P3 perm `euclidean` | 0.082 | 1.113 | 0.330 | **0.095** | **2.320** | 0.908 | 0.35 |
| C1 rot `kabsch` | 0.132 | 1.204 | 0.358 | 0.119 | 3.244 | 0.640 | **1.05** |
| C2 rot `heading` | 0.177 | 1.396 | 0.351 | 0.105 | 3.549 | 0.527 | **0.99** |
| C3 rot `heading` + matched prior | 0.129 | 1.164 | **0.309** | 0.098 | 3.087 | **0.420** | **1.03** |
| C4 rot `heading`, \(\kappa=2\) | 0.058 | 0.732 | 0.369 | 0.120 | 3.637 | 1.117 | −0.02 |
| F perm+rot `group_ot` | 0.144 | 1.328 | 0.355 | 0.100 | 2.306 | 0.582 | **1.02** |

### 9.1 The result that reorganizes the design

Read the last three columns together and the picture is unambiguous:

* **Steering gain splits cleanly by whether a rotation is applied.** ≈0 for `iid` and every
 permutation-only coupling; ≈1.0 for every `rot`/`perm_rot` variant. A \(\kappa=2\) dither
 destroys it entirely (C4 → −0.02), so soft alignment is not a free interpolation.
* **Source-orbit equivariance improves monotonically with coupling strength**, 1.22 → 0.42, a
 66% reduction for C3. The mechanism the toy study found does transfer.
* **Few-step error and straightness improve** — C3 is 16% better at one step, P3 is 38%
 straighter. This is the currency the few-step-policy line trades in (Consistency Policy RSS
 2024; OneDP ICML 2025; FlowPolicy AAAI 2025).
* **And distribution fidelity gets worse.** SWD 0.054 → 0.129, and, more diagnostically,
 *conditional* heading error nearly doubles: 0.677 → 1.164 rad.

The last two lines are not in tension by accident. A steering gain of 1 *means* the model has
learned to read the chunk's heading off the source phase. At inference the source phase is
drawn independently of the observation, so the heading is then chosen by chance — the marginal
stays plausible while the conditional collapses. **The property that makes DNO cheap is
exactly the property that hurts open-loop conditional accuracy.**

The unconditional double-ring problem could not exhibit this: with no observation to
condition on, there is no conditional accuracy to lose. It is a genuinely new failure mode
introduced by conditioning, and it is the strongest reason to run the toy-to-robot port
carefully rather than expecting Method C's numbers to reappear.

It also explains the earlier hypothesis correctly and rules out the wrong one: the
conditional-minibatch bias of §6.3 predicts the SWD loss for `perm` variants only, but the
worst SWD rows are the per-sample `rot` variants that have no minibatch bias at all. The
mechanism is delegation, not batch mixing.

### 9.2 Which means the coupling and DNO are one contribution, not two

If a policy delegates the heading to the noise, then **you must supply the heading**. A
one-dimensional orbit search over a task objective is precisely how you supply it, for the
cost of one batched forward pass. So the orbit-coupled policy is not meant to be run
open-loop from random noise; that configuration is the ablation, not the method.

Test 6 runs that 2×2 in miniature: \(K=16\) orbit angles, one batched forward pass, no
gradients, against a deliberately weak objective ("head toward the direction the observation
suggests", using the same noisy heading estimate the condition vector already carries — the
shape of a real geometric objective, not an oracle).

| method | steer gain | headMAE, no DNO | headMAE, +orbit | task \(\mathcal L\), no DNO | task \(\mathcal L\), +orbit | \(\Delta\mathcal L\) |
|---|---:|---:|---:|---:|---:|---:|
| A iid | −0.00 | 0.671 | 0.476 | −0.721 | −0.904 | −0.18 |
| P3 perm `euclidean` | 1.08 | 1.176 | 0.478 | −0.302 | −0.915 | −0.61 |
| C3 rot `heading` + prior | 1.02 | 1.164 | **0.436** | −0.287 | **−0.990** | **−0.70** |

The trade closes, and then some:

* After the orbit search, the **orbit-coupled policy has lower conditional heading error than
 the iid policy does** (0.436 vs 0.476) and reaches −0.990 against an objective whose optimum
 is −1.0. The open-loop deficit of §9.1 is not merely recovered, it is overturned.
* Per unit of compute the search is **3.8× more effective** on the coupled policy
 (\(\Delta\mathcal L\) −0.70 vs −0.18) at identical cost.
* The iid policy is not unaffected, and it is worth being precise about why: with steering
 gain 0, rotating the source produces \(K\) *unrelated* samples, so the orbit search
 degenerates into best-of-\(K\) resampling. That still helps a noisy objective. The claim is
 therefore not "search only works with the coupling" but the sharper and more defensible
 "**the coupling turns a blind best-of-\(K\) draw into a structured one-dimensional search over
 the coordinate that matters**".
* Caveat from the seeds: P3's steering gain is 1.08 here (seed 0) against a 3-seed mean of
 0.35 in test 5 — a symmetry-blind Euclidean OT coupling acquires steerability *sometimes*.
 Report steering gain per seed, not just its mean.

**Caveats.** Synthetic data, one small MLP, a fixed budget, and a task objective that is a
noisy proxy rather than a simulator. This validates the *machinery* and the *mechanism*, not
a LIBERO result.

---

## 10. Experiment matrix on LIBERO

| ID | Normalizer | Coupling | Inference prior | Purpose |
|---|---|---|---|---|
| P0 | per-dim min-max | iid | \(\mathcal N(0,I)\) | reproduce the repo baseline |
| P1 | SO(2) block | iid | \(\mathcal N(0,I)\) | **isolates the normalizer change** |
| P2 | SO(2) block | `perm`, `angle` | \(\mathcal N(0,I)\) | marginal-exact optimum |
| P3 | SO(2) block | `perm`, `euclidean` | \(\mathcal N(0,I)\) | **symmetry-blind OT control** |
| P4 | SO(2) block | `rot`, `heading` | \(\mathcal N(0,I)\) | direct toy transfer; measures the prior shift |
| P5 | SO(2) block | `rot`, `heading` | matched | prior shift removed |
| P6 | SO(2) block | `perm_rot`, `group_ot` | matched | strongest coupling (Klein-style) |
| P7 | SO(2) block | `rot`, `heading`, \(\kappa\in\{1,2,4,8\}\) | matched | the Pareto sweep replacing Round-2's \(p_{\mathrm{OT}}\) |

**P3 is not optional.** Without a symmetry-blind OT coupling in the table, any gain is
attributable to "some coupling helps" rather than to the symmetry in the coupling. That is
the first thing a reviewer asks, and OT-CFM / Immiscible Diffusion are the papers they will
name.

Optional P8: Eq.Bot-style observation canonicalization (arXiv:2511.15194), which already gives
non-equivariant policies SE(2) equivariance on LIBERO by wrapping them. It is the strongest
"why not just do X instead" and deserves either a head-to-head or an explicit argument (the
coupling is training-only, adds zero inference cost, and degrades gracefully when the symmetry
is broken; canonicalization enforces exact equivariance and does not).

**Metrics.**

* Success rate on LIBERO-10 at \(N_{\text{infer}}\in\{1,2,4,10\}\) — the low-\(N\) column is
 where the transport argument should show up.
* Validation FM loss and action-chunk MSE.
* Coupling diagnostics (logged every step): path length, angle gap, source heading \(R_1\),
 conditional velocity variance, coupling-active fraction.
* \(W_1(\text{source heading},\mathrm{Unif})\) — the prior-deformation axis.
* Source-orbit equivariance and steering gain, on trained checkpoints.
* Field equivariance evaluated on **four** spatial distributions — common iid interpolation
 points, each method's own training paths, each method's own inference trajectories, and
 off-manifold points. The toy Round-2 plan already calls for this, and it is what separates
 global relaxed equivariance from trajectory-local equivariance (recall Method D was better
 in the generator than in the common-path field metric).
* DNO: constraint satisfaction vs. iteration count, wall-clock, and success-rate delta, with
 the 2×2 of §8.2.

**Budget.** LIBERO rollouts are expensive; 3 seeds is the realistic floor (the toy used 10).
Report seed spread, and do not claim a difference that seed spread covers — the toy study
already had to soften a claim for exactly that reason (C vs E on SWD).

---

## 11. Risks, in order of how likely they are to bite

1. **Heading delegation** (§9.1). The largest effect measured, and the one that decides how
 the work is framed. Symptom: steering gain → 1, open-loop conditional error up, marginal
 quality roughly intact. Not fixable by regularization; it is the reason the orbit search is
 mandatory rather than optional. Diagnose with steering gain and `headMAE` together, and
 never report an orbit-coupled policy without its search.
2. **Prior shift** (§5). Mitigated by `perm` or by the matched prior; measured by source
 heading \(R_1\) and \(W_1\) to uniform. Do not skip the audit.
3. **Conditional minibatch bias** (§6.3). Shows up as: transport metrics improve, sample
 quality does not, *for `perm` variants only*. In the synthetic study it was **not** the
 dominant effect — the worst quality rows were the per-sample `rot` variants that have no
 minibatch bias at all — but it is still the right thing to sweep batch size for.
4. **Normalizer confound** (§3). If success drops between P0 and P1 for reasons unrelated to
 geometry (the bounded range of `limits` may matter for a transformer), the whole comparison
 is confounded. That is what P1 is for; use `vector_mode='quantile'` if bounded range turns
 out to matter.
5. **Rotation block is degenerate.** If \((\omega_x,\omega_y)\) carries a few percent of the
 energy, the coupling is translation-only. Diagnose with `vector_energy_fraction`; run the
 translation-only spec as the honest version.
6. **Stationary chunks.** LIBERO demos contain many near-static segments where the heading is
 meaningless. `min_heading_confidence` gates them; report what fraction it excludes.
7. **Wrong controller frame** (§2). Silently invalidates everything. Check the config.
8. **Wrong controller gain in the task losses.** `trans_gain` / `rot_gain` default to
 robosuite's `OSC_POSE` values (0.05 m, 0.5 rad); a wrong value quietly rescales every
 geometric constraint.
9. **Reward hacking in DNO.** The typicality regularizer and the trust region bound how far
 the search can drag the chunk off the demonstration manifold; they do not make a crude
 proxy correct. Report the fraction of DNO solutions that leave the trust region.
10. **Overclaiming equivariance** (§4). The RGB result is about noise-space structure. Sense
 (3) requires the state-obs ablation.

---

## 12. Relation to prior work, honestly

**Verified citations.**

| Work | Relation |
|---|---|
| Klein, Krämer, Noé, *Equivariant flow matching*, NeurIPS 2023, arXiv:2306.15030 | Nearest ancestor. Equivariant OT cost \(\tilde c=\min_g\|x_0-\rho(g)x_1\|^2\), Hungarian + Kabsch, applies the optimal \(g\). But the network is a fully equivariant EGNN, the domain is invariant-density molecular sampling, there is no conditioning and no policy. |
| Pooladian et al., *Multisample Flow Matching*, ICML 2023, arXiv:2304.14772 | The marginal-preservation template for any minibatch coupling. Cite for §5, §6.3. |
| Tong et al., OT-CFM, TMLR 2024; Liu et al., Rectified Flow, ICLR 2023 | Symmetry-blind coupling design. P3 is their coupling. |
| Li et al., *Immiscible Diffusion*, NeurIPS 2024, arXiv:2406.12303 | Assignment-based noise-data pairing for DDPM, and the exact argument §7 inherits — including its acknowledged looseness. |
| Wang et al., *Equivariant Diffusion Policy*, CoRL 2024, arXiv:2407.01812; Yang et al., *EquiBot*, CoRL 2024 | The architectural alternative. Explain why a training-time coupling is a different trade (zero inference cost, graceful under broken symmetry). |
| Hu et al., *3D Equivariant Visuomotor Policy Learning via Spherical Projection*, NeurIPS 2025 Spotlight, arXiv:2505.16969 | Justifies §4: a fixed camera image is invariant, not equivariant, under global scene rotation. |
| Deng et al., *Eq.Bot*, arXiv:2511.15194 (preprint) | **The dangerous comparison.** Already gives non-equivariant policies SE(2)/C₄ equivariance by canonicalization, on LIBERO. Cite and ablate. |
| Park, Chang, Choi, Horowitz, *Symmetry-Aware Steering of Equivariant Diffusion Policies*, arXiv:2512.11345 (preprint) | **The other dangerous comparison.** Equivariance-aware latent-noise steering of a diffusion policy. Differentiator: equivariant base + test-time RL only, versus non-equivariant base + training-time coupling + gradient DNO. State it in Related Work, not in rebuttal. |
| Wagenmaker, Nakamoto et al., DSRL, CoRL 2025, arXiv:2506.15799 | Noise-space RL on a frozen diffusion policy. The amortized alternative to §8. |
| Kang et al., *WarmPrior*, arXiv:2605.13959 (preprint) | The only other work changing a flow-matching robot policy's source. Temporal (mean-shift toward the previous chunk), not geometric, and it changes the source marginal. Cite as the closest source-side neighbour. |
| Karunratanakul et al., DNO, CVPR 2024, arXiv:2312.11994; Ben-Hamu et al., D-Flow, ICML 2024, arXiv:2402.14017; Yu et al., ADTG, CoRL 2024, arXiv:2410.10766 | The DNO lineage. |

**What appears new.**

1. Using a group-aligned coupling *specifically to induce structure in a deliberately
 non-equivariant policy*. Klein et al. align to shorten paths for an already-equivariant
 network; nobody frames alignment as a symmetry-injection mechanism.
2. The closed-form per-sample orbit rotation with a characterized induced source law, and
 the `MatchedHeadingPrior` resolution of the uniformity obstruction. \(O(B)\), no OT solve.
3. The obstruction itself: transport-vs-marginal as a lower bound, with the measured Pareto
 frontier. This is what the toy problem could not see.
4. Any symmetry-aware coupling on action-chunk manipulation policies.
5. The normalizer/irrep interaction (§3) — no prior analysis found.
6. Orbit-structured two-stage DNO, and the steering-gain metric that predicts when it works.
7. **The delegation trade-off** (§9.1): quantifying that noise-space controllability is
 acquired *at the expense of* observation-conditioned accuracy, and that a one-dimensional
 orbit search over a weak objective more than repays it. This only appears under
 conditioning, so no unconditional generative-modelling paper could have found it.

**What is not new on its own.** The noise-optimization half. It earns its place only as *the
thing the coupling makes better-conditioned* — which is a testable claim (§8.2), not a framing.

---

## 13. Staged plan

**Stage 0 — prerequisites (1 day).** Confirm the controller frame. Fit the SO(2) normalizer,
check `equivariance_error` ≈ 1e-7, report per-block energy fractions and the heading
histogram of LIBERO-10 chunks with \(W_1\) to uniform. If that \(W_1\) is near zero, the
obstruction of §5 does not bite on this dataset and `rot` is safe — a fact worth knowing
before running anything.

**Stage 1 — normalizer control (P0 vs P1).** No coupling. If P1 ≉ P0 in success rate,
understand why before proceeding; everything after this depends on it.

**Stage 2 — coupling comparison (P1–P6, 3 seeds).** Primary readout: success rate at
\(N\in\{1,2,4,10\}\). Secondary: transport diagnostics, source-orbit equivariance, steering
gain. P3 is the control that gives the symmetry claim its meaning.

**Stage 3 — the \(\kappa\) Pareto sweep (P7).** Two axes: transport reduction against prior
deformation. This replaces the toy's planned soft-radial \(p_{\mathrm{OT}}\) sweep, which is
the wrong knob for non-uniform target headings.

**Stage 4 — DNO (the 2×2 of §8.2).** Tier-0 objectives first, stage-1-only before stage-2.
Report wall-clock next to every number, and report \(\Delta\mathcal L\) *per forward pass*
rather than only the final value — the whole claim is about efficiency of search, and the
iid baseline gets a real (if smaller) benefit from best-of-\(K\) resampling.

Because of the delegation contract (§8.3), stages 2 and 4 must be read together: a coupling
that loses open-loop success in stage 2 has not failed until it also fails to recover it in
stage 4. Plan the compute accordingly — do not kill a coupling on stage-2 numbers alone.

**Stage 5 — the state-obs ablation.** A LIBERO variant with state-only observations, where
the observation *can* be rotated, so that sense-(3) equivariance is measurable and the
relaxed-equivariance claim can be made in its strong form on at least one setting.

---

## 14. Compact summary

\[
\boxed{
\begin{aligned}
\text{normalizer}&: \text{per-dim min-max}\rightarrow\text{block-isotropic, zero offset on vector blocks (required first)},\\
\text{coupling}&: \texttt{perm}\ \text{marginal-exact but bounded by }W_1(\mathrm{Unif},p_\theta);\
\texttt{rot}\ \text{unbounded but shifts the prior},\\
\text{prior}&: \texttt{rot,align=heading} + \text{matched inference prior} \Rightarrow \text{no mismatch, full alignment},\\
\text{equivariance}&: \text{RGB gives \emph{noise-space} equivariance; sense-(1) needs the state-obs ablation},\\
\text{diffusion}&: \texttt{perm}\ \text{only; \texttt{rot} is unsound for DDPM},\\
\text{delegation}&: \text{steer gain}\uparrow \Leftrightarrow \text{heading read off the noise} \Rightarrow \text{open-loop conditional accuracy}\downarrow,\\
\text{DNO}&: \text{orbit grid (1-D, exhaustive)} \rightarrow \text{IrrepAdam residual descent, and it repays the delegation},\\
\text{key test}&: \{\text{iid},\text{orbit}\}\times\{\text{stage 1 on},\text{off}\};\ \text{measured } 3.8\times\ \Delta\mathcal L\ \text{per forward pass}.
\end{aligned}}
\]

The single sentence to defend: *an orbit-aligned source-target coupling gives a vanilla
flow-matching action-chunk policy a group-structured noise space at zero inference cost, and
that structure is what makes a one-dimensional test-time search over the group orbit both
necessary — the policy now delegates its heading to the noise — and cheap enough for
closed-loop control.*
