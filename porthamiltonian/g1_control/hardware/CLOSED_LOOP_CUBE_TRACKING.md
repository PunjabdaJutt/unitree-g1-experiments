# Closed-loop cube tracking pipeline (design doc, not implemented)

Status: **not started**. This extends `PIPELINE.md`'s Stage 4 ("camera-driven
cube detection onto real hardware") specifically for the case where the cube
position needs to be *re-detected and re-planned continuously* during the
approach, rather than detected once and committed to. Written before
implementation so the design can be reviewed/corrected first -- do not build
against this until it's been checked against real bench data (frame rate,
detection jitter, servo response), see "Open questions" at the bottom.

## Where things stand today (on-demand replan only)

- `dry_run_grasp.py` / the real grasp loop take **one** cube detection
  (`localize_3d.mask_to_3d_point` -> `loc.point_base`), call
  `trajectory_gen.generate_grasp_waypoints()` **once**, and solve IK for the
  resulting `pre_grasp -> grasp -> post_grasp` waypoints. If the detection was
  wrong, or the cube moves, or the robot's own base sways between detection
  and arrival, nothing corrects for it.
- `trajectory_gen.GraspTrajectory` (added alongside this doc) is a small
  stateful wrapper that lets you push a **new** `p_cube_base` and regenerate
  all three waypoints from it on demand -- e.g. call `.update(new_pos)` once
  right before the arm starts moving, or after a re-detection you trust more.
  It does **not** poll perception itself and nothing calls it repeatedly yet.

This doc is about closing that gap: continuously feeding fresh detections
into the control loop while the arm is actually moving.

## Proposed pipeline

```
D435 camera  -->  CubeSegmenter.detect()  -->  mask_to_3d_point()
(camera_grabber /     (Detection/)              (localize_3d.py)
 detection.py)              |                         |
                             v                         v
                     color+mask, ~camera fps    point_base (pelvis frame)
                                                  + confidence
                                                        |
                                                        v
                                          +-------------------------+
                                          |  filter + staleness gate |
                                          +-------------------------+
                                                        |
                                                        v
                                         GraspTrajectory.update(p_filtered)
                                                        |
                                                        v
                                          ik_two_hands_dls(q0=last arm q)
                                                        |
                                                        v
                                             Custom.arm_q_target (rt/arm_sdk)
```

1. **Perception producer, independent rate.** The camera/detector runs at its
   own rate (D435 RGB-D + `CubeSegmenter` inference latency), which will be
   slower and less regular than the control loop's `control_dt` (0.02s / 50Hz
   in `g1_hw_ph_finger_control.py`). Don't try to force it onto the control
   tick -- run it as a producer (thread or separate process) that just
   publishes its latest `(point_base, confidence, timestamp)` result whenever
   it has one.

2. **Latest-value handoff, not a queue you drain.** The control tick only
   ever wants the *most recent* cube estimate, not a backlog of stale ones --
   a lock-guarded "latest reading + timestamp" (or a 1-deep queue that
   overwrites) is enough. The control loop reads it non-blockingly each tick;
   if nothing new has arrived, it just keeps using the last accepted value.

3. **Filter before it reaches `GraspTrajectory`.** Raw per-frame
   `point_base` will jitter (pixel noise, depth noise, segmentation mask
   boundary noise). Feed it through a simple filter (EMA to start; a small
   Kalman filter if EMA lag turns out to matter) *before* calling
   `GraspTrajectory.update()`, so IK isn't re-solved against noise every
   frame. The right time constant is a bench-measurement question (see below).

4. **Staleness / confidence gate.** If `confidence < MIN_CONFIDENCE`
   (`dry_run_grasp.py` already uses `0.5` as this threshold) or the last
   accepted reading is older than some timeout, do **not** feed it into
   `GraspTrajectory` -- hold the last trusted target instead. A perception
   process that has died or lost the cube (e.g. occluded by the hand during
   final approach) must not silently mean "assume the cube didn't move."

5. **Rate/distance clamp on accepted updates.** Cap how far a single accepted
   update can move `p_cube_base` from the previous accepted value (e.g. some
   cm per tick). This is a cheap guard against a single bad detection
   snapping the arm target across the workspace; repeated clamp-triggering is
   itself a signal to fall back to "hold last target" or abort.

6. **Only track during approach, freeze before contact.** Tracking should
   apply while the arm is still closing distance (a `pre_grasp`/"approach"
   type segment) and must **stop** once the hand is committed to the final
   grasp -- i.e. right around when `g1_hw_ph_finger_control.py` would switch
   from its position-mode `approach_close` segment into `grasp_hold`
   (force mode). Reasons: (a) the hand itself will occlude the camera's view
   of the cube during final approach, so late detections are the least
   trustworthy exactly when you'd be most tempted to use them; (b) continuing
   to nudge the Cartesian target while fingers are actively closing risks
   dragging the grasp instead of holding it. Concretely this likely means a
   new segment concept (e.g. `arm_mode="tracking"` alongside the existing
   `hand_mode` states) that is only legal during the pre-contact segments.

7. **Re-solve IK incrementally, not from rest.** Each accepted update should
   call `ik_two_hands_dls(..., q0=<current commanded arm q>)`, the same
   pattern `Custom._load_segment()` already uses at segment boundaries --
   never `q0=zeros(17)` (that's `dry_run_grasp.py`'s explicit "assumed rest
   pose" caveat, fine for an offline feasibility check, wrong for a live
   loop). This keeps each re-plan a small IK step from wherever the arm
   already is, rather than a fresh global solve.

## Where this hooks into `g1_hw_ph_finger_control.py`

`Custom._tick()` currently computes `q_arm_cmd` by linearly interpolating
between `arm_q_start` and a `Segment`'s fixed `arm_q_target` over `seg.T`.
Continuous tracking would replace that interpolation, for segments flagged as
trackable, with: read latest filtered+gated cube estimate -> if accepted,
`GraspTrajectory.update()` -> `ik_two_hands_dls(q0=current arm_q_target)` ->
new `arm_q_target`. Everything downstream (arm_sdk publish, weight ramp,
safety gates) stays the same.

## Open questions to resolve on the bench before implementing

- Real camera+detector loop rate and latency (drives whether EMA is enough or
  a proper Kalman filter is needed, and what "stale" should mean in seconds).
- Frame-to-frame jitter magnitude in `point_base` under real lighting/cube
  conditions (sizes the filter time constant and the distance clamp in step 5).
- How much visible jerk the arm's position-PD (`arm_kp=60`, `arm_kd=1.5`)
  produces from small target updates at the chosen re-plan rate -- may need
  to throttle re-plans to every N control ticks rather than every accepted
  perception update.
- Exact segment/geometric criterion for "stop tracking, commit to grasp"
  (distance threshold? time-in-segment? first successful force-mode contact?)
  -- currently just sketched in step 6 above, needs a concrete rule.
