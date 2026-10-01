# trajectory_gen.py
"""Cube position (pelvis frame) -> pre-grasp / grasp / post-grasp waypoints.

Grasp strategy: top-down approach along the (assumed horizontal) surface
normal, default [0, 0, 1] (pelvis frame). Waypoints:

    p_pre  = p_cube + approach_dist * normal   (hover above the cube)
    p_grasp = p_cube + grasp_depth  * normal   (at/slightly into the cube)
    p_post  = p_cube + lift_height  * normal   (lift straight back up)

Orientation: the hand's approach ("plunge") axis is aligned with -normal
(pointing down into the surface) at every waypoint -- same orientation
throughout, only position changes, which is the simplest correct choice for
a top-down grasp. Rotation matrices (not quaternions) are the primary
output here because ph.kinematics.ik_two_hands_dls's RR_target parameter
takes a 3x3 rotation matrix directly; rotation_matrix_to_quat() is provided
as a convenience if you need (x,y,z,qx,qy,qz,qw) for something else.

wrist_yaw_link -> palm convention (resolved 2026-09-04, see
_build_approach_rotation()): right_wrist_yaw_link is NOT rigidly aligned
with the hand -- R_base_link_joint (URDF) attaches R_hand_base_link to it
via a fixed rpy=(pi, 0, -pi/2), so the old identity assumption (wrist
local Z = plunge axis) was wrong. Derived FINGER_AXIS_LOCAL/PALM_NORMAL_LOCAL
below the same way g1_hw_arm_hand_test.py's default_segments() comment
derived a one-off wrist_roll value for a single pose (cross of the finger
direction with the thumb-to-pinky direction, via ph.fk_fingers) -- but
computed here as vectors in right_wrist_yaw_link's OWN local frame, which
fk_chain/fk_fingers confirm are exactly pose-independent (verified
numerically to ~1e-16 across three unrelated arm configurations), so they
only need computing once and work for any desired approach normal, not just
one bench-tuned pose.
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PH_SUBMODULE_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 3)))
if _PH_SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, _PH_SUBMODULE_ROOT)
from scripts.g1_control.ph import fk_chain, default_urdf, IK17_JOINT_NAMES, ik_two_hands_dls  # noqa: E402
from scripts.g1_control.ph.fk_fingers import fk_fingers, R_BASE_XYZ, R_BASE_RPY  # noqa: E402
from scripts.g1_control.ph.spatial import rpy2R  # noqa: E402

DEFAULT_NORMAL = np.array([0.0, 0.0, 1.0])
DEFAULT_APPROACH_DIST = 0.12   # m above the cube for the pre-grasp hover
DEFAULT_LIFT_HEIGHT = 0.15     # m above the cube for the post-grasp lift
DEFAULT_GRASP_DEPTH = 0.0      # m along -normal from the cube center; TODO tune vs.
                                # hand geometry (0 = close exactly at the cube center)

# A reference "forward" direction (pelvis frame) used only to fix the
# approach rotation's remaining degree of freedom (rotation about the
# approach/plunge axis itself, i.e. where the fingers point within the
# plane perpendicular to the plunge direction). +X (robot forward) is a
# reasonable default for a cube presented in front of the robot; change if
# your grasp approaches from the side.
REFERENCE_FORWARD = np.array([1.0, 0.0, 0.0])


def _hand_reference_vectors_local():
    """(finger_axis, palm_normal), unit vectors in right_wrist_yaw_link's
    OWN local frame -- constant regardless of arm pose (see module
    docstring), computed once here from the URDF+finger-chain geometry via
    fk_chain/fk_fingers rather than hand-transcribed, per this repo's
    "single source of truth" convention (see fk_fingers.py's own
    docstring). Uses an all-fingers-open reference hand pose (q_hand6=0);
    palm_normal is only weakly sensitive to finger curl since it's mostly
    determined by R_hand_base_link's rigid mounting, so any reasonable
    reference curl would do."""
    urdf = default_urdf()
    q17 = dict(zip(IK17_JOINT_NAMES, [0.0] * 17))
    theta29 = np.zeros(29)  # arm at rest, all fingers open -- pose is irrelevant, see above

    T_wrist = fk_chain(urdf, "pelvis", "right_wrist_yaw_link", q17)
    T_hand_base = T_wrist @ _T(rpy2R(R_BASE_RPY), R_BASE_XYZ)
    tips = {f: T[:3, 3] for f, T in fk_fingers(theta29).T_R.items()}

    hb_origin = T_hand_base[:3, 3]
    avg_four = np.mean([tips["index"], tips["middle"], tips["ring"], tips["pinky"]], axis=0)
    finger_axis = _unit(avg_four - hb_origin)
    thumb_to_pinky = _unit(tips["pinky"] - tips["thumb"])
    palm_normal = _unit(np.cross(finger_axis, thumb_to_pinky))

    R_wrist_inv = T_wrist[:3, :3].T  # rotation matrices are orthogonal: inverse = transpose
    return R_wrist_inv @ finger_axis, R_wrist_inv @ palm_normal


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


FINGER_AXIS_LOCAL, PALM_NORMAL_LOCAL = _hand_reference_vectors_local()


@dataclass
class Waypoint:
    name: str
    position: np.ndarray   # (3,) meters, pelvis frame
    R: np.ndarray           # (3,3) rotation matrix, pelvis frame


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (qx, qy, qz, qw)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw])


def _orthonormal_frame(axis3: np.ndarray, ref1: np.ndarray) -> np.ndarray:
    """3x3 [col1, col2, col3] with col3=axis3 (unit) and col1 the
    Gram-Schmidt projection of ref1 onto axis3's orthogonal complement,
    col2 = cross(col3, col1). Falls back to a different ref if ref1 is
    (near-)parallel to axis3, so the projection doesn't divide by ~0."""
    a3 = _unit(axis3)
    proj = ref1 - np.dot(ref1, a3) * a3
    if np.linalg.norm(proj) < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0])
        proj = fallback - np.dot(fallback, a3) * a3
    a1 = _unit(proj)
    a2 = np.cross(a3, a1)
    return np.column_stack([a1, a2, a3])


def _build_approach_rotation(normal: np.ndarray, forward_ref: np.ndarray) -> np.ndarray:
    """right_wrist_yaw_link's target orientation (pelvis frame, 3x3) such
    that the Inspire hand's palm faces -normal (plunging into the surface)
    with its fingers pointing as close to forward_ref as geometry allows.

    Solved by aligning two orthonormal frames sharing the same 3rd axis
    convention (palm normal / desired plunge direction): the hand's own
    fixed reference frame S = [finger_axis, ., palm_normal], both in
    right_wrist_yaw_link's local coordinates (FINGER_AXIS_LOCAL /
    PALM_NORMAL_LOCAL, module-level, URDF-derived -- see
    _hand_reference_vectors_local()), and the desired frame D =
    [forward_ref, ., -normal] in pelvis coordinates. Since both are
    orthonormal, the rotation mapping S's columns onto D's column-for-column
    is R = D @ S.T (S^-1 = S.T for an orthonormal matrix) -- this is exactly
    the wrist_yaw_link orientation ik_two_hands_dls's RR_target needs, NOT
    an orientation for the hand/palm frame itself.
    """
    D = _orthonormal_frame(-normal, forward_ref)                    # pelvis frame: palm normal, forward
    S = _orthonormal_frame(PALM_NORMAL_LOCAL, FINGER_AXIS_LOCAL)     # wrist-local: palm normal, fingers
    return D @ S.T


def generate_grasp_waypoints(
    p_cube_base: np.ndarray,
    normal: Optional[np.ndarray] = None,
    approach_dist: float = DEFAULT_APPROACH_DIST,
    lift_height: float = DEFAULT_LIFT_HEIGHT,
    grasp_depth: float = DEFAULT_GRASP_DEPTH,
    hand_offset: Optional[np.ndarray] = None,
) -> List[Waypoint]:
    """hand_offset: optional (3,) meters, pelvis frame, constant offset from
    the cube-center grasp point to the wrist target IK actually solves for
    (i.e. fingertip-to-wrist offset) -- TODO measure from the Inspire hand's
    geometry / URDF fingertip-to-wrist_yaw_link offset; zero (IK targets the
    cube center directly) is very likely wrong once fingers are involved."""
    normal = DEFAULT_NORMAL if normal is None else np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    offset = np.array([0.0, 0.0, 0.05]) if hand_offset is None else np.asarray(hand_offset, dtype=float)
    p_cube_base = np.asarray(p_cube_base, dtype=float)

    R = _build_approach_rotation(normal, REFERENCE_FORWARD)

    p_pre = p_cube_base + approach_dist * normal + offset
    p_grasp = p_cube_base - grasp_depth * normal + offset
    p_post = p_cube_base + lift_height * normal + offset

    return [
        Waypoint("pre_grasp", p_pre, R),
        Waypoint("grasp", p_grasp, R),
        Waypoint("post_grasp", p_post, R),
    ]


class GraspTrajectory:
    """Stateful wrapper around generate_grasp_waypoints() for on-demand
    replanning: holds the grasp shape (normal, approach_dist, lift_height,
    grasp_depth, hand_offset) fixed and lets the caller push a freshly
    re-detected cube position via update() to regenerate all three waypoints
    from it. Does NOT poll perception itself -- call update() whenever you
    have a new p_cube_base you trust (e.g. a fresh detection, right before
    the arm starts moving). See CLOSED_LOOP_CUBE_TRACKING.md for how this
    fits into a future continuous (every-tick) tracking pipeline."""

    def __init__(
        self,
        p_cube_base: np.ndarray,
        normal: Optional[np.ndarray] = None,
        approach_dist: float = DEFAULT_APPROACH_DIST,
        lift_height: float = DEFAULT_LIFT_HEIGHT,
        grasp_depth: float = DEFAULT_GRASP_DEPTH,
        hand_offset: Optional[np.ndarray] = None,
    ):
        self.normal = normal
        self.approach_dist = approach_dist
        self.lift_height = lift_height
        self.grasp_depth = grasp_depth
        self.hand_offset = hand_offset
        self.p_cube_base: np.ndarray = np.asarray(p_cube_base, dtype=float)
        self.waypoints: List[Waypoint] = []
        self.update(self.p_cube_base)

    def update(self, new_p_cube_base: np.ndarray) -> List[Waypoint]:
        """Re-run generate_grasp_waypoints() from a freshly re-detected cube
        position, keeping every other grasp parameter fixed. Stores the
        result on self.waypoints (and self.p_cube_base) and returns it."""
        self.p_cube_base = np.asarray(new_p_cube_base, dtype=float)
        self.waypoints = generate_grasp_waypoints(
            self.p_cube_base,
            normal=self.normal,
            approach_dist=self.approach_dist,
            lift_height=self.lift_height,
            grasp_depth=self.grasp_depth,
            hand_offset=self.hand_offset,
        )
        return self.waypoints


def solve_ik_step(tracker: GraspTrajectory, urdf, pL_hold: np.ndarray, q_prev: np.ndarray,
                   target_name: str = "pre_grasp"):
    """One tracking-loop step against `tracker`'s CURRENT waypoints -- call
    `tracker.update(new_p_cube_base)` first if you have a fresh detection to
    feed in. Solves IK for the `target_name` waypoint warm-started from
    `q_prev` (not from rest -- see CLOSED_LOOP_CUBE_TRACKING.md step 7), then
    re-runs FK on the solved joints to verify where the wrist actually ends
    up vs. the IK target. Shared by dry_run_grasp.py's --live mode and
    Detection/test_perception_only.py's --track mode so both draw from the
    same logic instead of duplicating it.

    Returns (waypoint, q_new, ik_info, wrist_fk_pos, fk_err_m)."""
    wp = next(w for w in tracker.waypoints if w.name == target_name)
    q_new, info = ik_two_hands_dls(pL_target=pL_hold, pR_target=wp.position,
                                    RR_target=wp.R, q0=q_prev, urdf=urdf)
    wrist_fk = fk_chain(urdf, "pelvis", "right_wrist_yaw_link",
                         dict(zip(IK17_JOINT_NAMES, q_new)))[:3, 3]
    fk_err = float(np.linalg.norm(wrist_fk - wp.position))
    return wp, q_new, info, wrist_fk, fk_err
