# Modified DH Parameter Tables — Unitree G1 Inspire-Hand DFQ

Convention: Craig MDH (1989) `T_{i-1→i} = Rx(α_{i-1}) Tx(a_{i-1}) Rz(θ_i + θ_off) Tz(d_i)`

Base frame: `{L|R}_hand_base_link` (attached at wrist yaw + fixed URDF offset)

Units: `a`, `d` in mm; `α`, `θ_off`, `q_lo`, `q_hi` in degrees. `Ax` = sign(axis_z).
`Type`: `act` = actuated, `1.6x`/`2.4x`/`1.0x` = mimic joint with that multiplier on the
parent actuated joint.

## Left Hand

### Thumb

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | L_thumb_prox_yaw | -16.96 | 90 | 69.10 | 99.71 | +1 | -5.73 | 74.49 | act |
| 1 | L_thumb_prox_pitch | 11.50 | 90 | -8.00 | 38.10 | -1 | -5.73 | 34.38 | act |
| 2 | L_thumb_inter | 56.00 | 0 | 0.80 | -11.46 | -1 | -5.73 | 34.38 | 1.6x |
| 3 | L_thumb_dist | 22.65 | 0 | 1.20 | 63.36 | -1 | -5.73 | 34.38 | 2.4x |

### Index

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | L_index_prox | 0.29 | 178 | -27.48 | 94.32 | -1 | 0 | 97.40 | act |
| 1 | L_index_inter | 32.13 | 0 | 1.00 | 85.68 | -1 | 0 | 97.40 | 1.0x |

### Middle

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | L_middle_prox | 137.10 | 180 | -12.95 | 4.44 | -1 | 0 | 97.40 | act |
| 1 | L_middle_inter | 32.13 | 0 | 1.00 | 176.40 | -1 | 0 | 97.40 | 1.0x |

### Ring

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | L_ring_prox | -0.29 | 176.99 | -0.89 | -85.68 | -1 | 0 | 97.40 | act |
| 1 | L_ring_inter | 32.13 | 0 | 1.00 | -94.32 | -1 | 0 | 97.40 | 1.0x |

### Pinky

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | L_pinky_prox | -0.29 | 174 | 11.16 | -85.68 | -1 | 0 | 97.40 | act |
| 1 | L_pinky_inter | 32.13 | 0 | 1.00 | -94.32 | -1 | 0 | 97.40 | 1.0x |

## Right Hand

### Thumb

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R_thumb_prox_yaw | 16.96 | 90 | -69.10 | 106.21 | -1 | -5.73 | 74.49 | act |
| 1 | R_thumb_prox_pitch | -11.50 | 90 | 8.00 | -141.90 | +1 | -5.73 | 34.38 | act |
| 2 | R_thumb_inter | 56.00 | 0 | -0.80 | -11.46 | +1 | -5.73 | 34.38 | 1.6x |
| 3 | R_thumb_dist | 22.65 | 0 | -1.20 | -116.64 | +1 | -5.73 | 34.38 | 2.4x |

### Index

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R_index_prox | -0.29 | 178 | 27.49 | -85.33 | +1 | 0 | 97.40 | act |
| 1 | R_index_inter | 32.13 | 0 | -1.00 | -94.67 | +1 | 0 | 97.40 | 1.0x |

### Middle

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R_middle_prox | 0.29 | 180 | 12.95 | 94.32 | +1 | 0 | 97.40 | act |
| 1 | R_middle_inter | 32.13 | 0 | -1.00 | 85.68 | +1 | 0 | 97.40 | 1.0x |

### Ring

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R_ring_prox | 0.29 | 176.99 | 0.89 | 94.32 | +1 | 0 | 97.40 | act |
| 1 | R_ring_inter | 32.13 | 0 | -1.00 | 85.68 | +1 | 0 | 97.40 | 1.0x |

### Pinky

| # | Joint | a | α | d | θ_off | Ax | q_lo | q_hi | Type |
|---|---|---|---|---|---|---|---|---|---|
| 0 | R_pinky_prox | 0.29 | 174 | -11.17 | 94.32 | +1 | 0 | 97.40 | act |
| 1 | R_pinky_inter | 32.13 | 0 | -1.00 | 85.68 | +1 | 0 | 97.40 | 1.0x |

## Notes

- The MDH convention used is Craig (1989): the `α` and `a` parameters describe the *previous*
  link, so frame `i` is built from the `i`-th joint.
- `α ≈ 90°` on both thumb joints indicates the thumb rotates *out of the finger plane*
  (abduction/adduction + flexion).
- `α ≈ 180°` on non-thumb proximal joints reflects the URDF convention that all proximal
  joints rotate in a direction antiparallel to the hand-base z-axis.
- The `a = 137.10 mm` entry for L/R Middle proximal encodes the **hand-base to proximal-joint
  Y-offset**; all other non-thumb proximal `a` values (±0.29 mm) are effectively zero
  (numerical rounding of the common-perpendicular algorithm for nearly-parallel axes).
- Mimic joints are not independently actuated: their joint angle is set by the URDF `<mimic>`
  tag to a fixed multiple of the parent proximal joint. The scale factors are 1.0× (index/
  middle/ring/pinky intermediate), 1.6× (thumb intermediate), and 2.4× (thumb distal).
- All MDH rows satisfy self-consistency error < 4×10⁻⁹ rad/m (float64 rounding). The
  URDF-chain FK (`fk_dh`) matches `fk_fingers` to < 0.001 mm over 200 random configurations.

See `analytical_ik_fingers.md` in this same directory for why the MDH table above is used only
for inspection/printing, never for FK computation — that's the 137mm-offset limitation.
