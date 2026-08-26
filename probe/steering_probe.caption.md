Orbit steering probe on LIBERO-10 checkpoints (epochs P1 iid (control)=1000, P4 rot/heading (coupled)=786  (NOT epoch-matched)).

For a frozen policy F, a fixed batch of 64 validation observations and a fixed source noise z, the source is rotated by rho(phi) over 24 angles spanning (0, 2pi) and the change in the generated chunk's heading is recorded: dtheta(phi) = wrap(heading(F(rho(phi)z | o)) - heading(F(z | o))). Each arm contributes 1536 points.

Arms:
  - P1 iid (control): coupling mode=iid, prior=standard, 10-step sampler; gain -0.000, phase consistency 0.030; low-confidence points 0.0%
  - P4 rot/heading (coupled): coupling mode=rot, align=heading, prior=standard, 10-step sampler; gain +1.010, phase consistency 0.674; low-confidence points 0.8%

Readout spec forced identically across arms: translation_only (action_dim=7;vector_blocks=((0, 1), (3, 4));scalar_dims=(2, 5, 6);block_weights=(1.0, 0.0)). Each checkpoint's own trained action_spec is deliberately NOT used: chunk_heading weights the vector blocks, so arms trained before and after the G4' decision would otherwise report different quantities. rotate_chunk acts on vector_blocks and rotates every block regardless of the weights, so forcing the readout changes what is measured, never what is done to the noise.

Identical across arms: observations (obs_seed=1234), source noise (z_seed=5678), readout spec. The checkpoint is the only difference.

The dashed reference breaks at phi = pi because dtheta is wrapped to (-pi, pi]; it is not a discontinuity in the policy. Faded points are those whose chunk-heading confidence falls below 0.05 -- a chunk whose planar deltas cancel over the horizon has no meaningful heading. They are shown rather than dropped because how many there are is part of the result.

Measured on validation observations only: no rollouts, no environment. The probe is therefore independent of task success and of the operating point.

This figure shows that the steering channel exists. It does NOT show that using it improves task success (that is Phase 6's on-benchmark 2x2), it says nothing about the transport claim (straightness / few_step_gap are separate metrics), and it is noise-space equivariance only -- the observation is held fixed while the noise rotates, so it is not equivariance in the textbook sense.
