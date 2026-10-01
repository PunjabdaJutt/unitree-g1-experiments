# pin_kinematics_client.py
""".venv-side client for pin_kinematics_server.py -- run the server in its
own terminal under the conda `pinocchio` env first (see that file's own
docstring), then call ik_two_hands_dls_remote()/fk_chain_remote() from
here, from ordinary .venv code (grasp_ph.py, dry_run_grasp.py, a REPL,
anything). Only needs numpy + the stdlib (socket/json/struct) -- does NOT
import pinocchio or pin_kinematics.py itself, so importing this file never
risks pulling .venv's broken `pin` package into the picture (see
pin_kinematics.py's own docstring for that bug). Safe to import
unconditionally; only actually touches the network (well, a local Unix
socket) when you call one of the functions below, and raises a clear
ConnectionRefusedError-derived message (not a cryptic OSError) if the
server isn't running yet.

ik_two_hands_dls_remote() has the SAME signature and (q, IKInfo) return
shape as kinematics.ik_two_hands_dls -- same field names on IKInfo too --
so it's a drop-in swap at any existing call site: change the import, not
the call.
"""
import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Deliberately NOT `from pin_kinematics_server import SOCK_PATH`: that module's
# own top-level imports pull in pin_kinematics.py -> `import pinocchio`, which
# is exactly the broken import this client exists to avoid in .venv (see this
# file's own docstring). One constant, duplicated rather than shared -- keep
# in sync with pin_kinematics_server.py's own SOCK_PATH if it ever changes.
SOCK_PATH = "/tmp/g1_pin_kinematics.sock"


class KinematicsServerNotRunning(ConnectionError):
    def __init__(self):
        super().__init__(
            f"Can't reach the Pinocchio kinematics server at {SOCK_PATH!r}. Start it first, in its own "
            "terminal, under the conda `pinocchio` env:\n"
            "    /home/ammar/miniconda3/envs/pinocchio/bin/python3 "
            "scripts/g1_control/ph/pin_kinematics_server.py\n"
            "(plain `conda activate pinocchio` can still resolve the wrong python if that shell already "
            "sourced .venv -- see pin_kinematics.py's own docstring on this -- the full interpreter path "
            "above sidesteps it.)"
        )


@dataclass
class IKInfo:
    converged: bool = False
    iters: int = 0
    final_errors: Dict[str, float] = field(default_factory=dict)


def _request(payload: dict) -> dict:
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(SOCK_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        raise KinematicsServerNotRunning() from e
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.sendall(struct.pack(">I", len(body)) + body)
        header = _recv_exact(conn, 4)
        (length,) = struct.unpack(">I", header)
        resp = json.loads(_recv_exact(conn, length).decode("utf-8"))
    finally:
        conn.close()
    if "error" in resp:
        raise RuntimeError(f"pin_kinematics_server error: {resp['error']}")
    return resp


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection before sending a full response")
        buf += chunk
    return buf


def ping() -> bool:
    """True if the server is reachable and responding."""
    try:
        return _request({"cmd": "ping"}).get("ok", False)
    except KinematicsServerNotRunning:
        return False


def fk_chain_remote(q17: np.ndarray, frame_name: str) -> np.ndarray:
    resp = _request({"cmd": "fk", "q17": np.asarray(q17, dtype=float).tolist(), "frame": frame_name})
    return np.array(resp["T"])


def ik_two_hands_dls_remote(
    pL_target: np.ndarray, pR_target: np.ndarray,
    RL_target: Optional[np.ndarray] = None, RR_target: Optional[np.ndarray] = None,
    use_waist: bool = True, lock_joint_names: Optional[Sequence[str]] = None,
    waist_weight: float = 1.0, wrist_weight: float = 1.0,
    joint_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    q0: Optional[np.ndarray] = None,
    max_iters: int = 200, lamb: float = 0.05, alpha: float = 0.8,
    tol_p: float = 1e-4, tol_w: float = 1e-4, dq_max: Optional[float] = None,
    w_pos: float = 1.0, w_rot: float = 1.0,
) -> Tuple[np.ndarray, IKInfo]:
    """Drop-in remote equivalent of kinematics.ik_two_hands_dls (and its
    Pinocchio-native counterpart, pin_kinematics.ik_two_hands_dls_pin) --
    identical arguments, identical (q, IKInfo) return shape. The actual
    solve runs in pin_kinematics_server.py's process; this function only
    serializes the request and deserializes the response."""
    payload = {
        "cmd": "ik_two_hands_dls",
        "pL_target": np.asarray(pL_target, dtype=float).tolist(),
        "pR_target": np.asarray(pR_target, dtype=float).tolist(),
        "RL_target": np.asarray(RL_target, dtype=float).tolist() if RL_target is not None else None,
        "RR_target": np.asarray(RR_target, dtype=float).tolist() if RR_target is not None else None,
        "use_waist": use_waist,
        "lock_joint_names": list(lock_joint_names) if lock_joint_names else None,
        "waist_weight": waist_weight, "wrist_weight": wrist_weight,
        "joint_bounds": {k: list(v) for k, v in joint_bounds.items()} if joint_bounds else None,
        "q0": np.asarray(q0, dtype=float).tolist() if q0 is not None else None,
        "max_iters": max_iters, "lamb": lamb, "alpha": alpha,
        "tol_p": tol_p, "tol_w": tol_w, "dq_max": dq_max,
        "w_pos": w_pos, "w_rot": w_rot,
    }
    resp = _request(payload)
    q = np.array(resp["q"])
    info = IKInfo(converged=resp["converged"], iters=resp["iters"], final_errors=resp["final_errors"])
    return q, info


def shoulder_origin_remote(waist_q: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """(3,) position of the right shoulder pivot in pelvis coordinates --
    see pin_kinematics.shoulder_origin_pin's own docstring (same function,
    run remotely)."""
    resp = _request({"cmd": "shoulder_origin", "waist_q": list(waist_q)})
    return np.array(resp["t"])


def fk_right_arm_remote(q7: np.ndarray, waist_q: Sequence[float] = (0.0, 0.0, 0.0),
                         frame_name: str = "right_wrist_yaw_link") -> np.ndarray:
    """(4,4) T_frame_from_shoulder for the 7-DOF right arm only -- see
    pin_kinematics.fk_right_arm_pin's own docstring (same function, run
    remotely)."""
    resp = _request({"cmd": "fk_right_arm", "q7": np.asarray(q7, dtype=float).tolist(),
                      "waist_q": list(waist_q), "frame": frame_name})
    return np.array(resp["T"])


def ik_right_arm_dls_remote(
    p_target: np.ndarray, R_target: Optional[np.ndarray] = None,
    waist_q: Sequence[float] = (0.0, 0.0, 0.0),
    q0_7: Optional[np.ndarray] = None,
    joint_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    waist_weight: float = 1.0, wrist_weight: float = 1.0,
    max_iters: int = 200, lamb: float = 0.05, alpha: float = 0.8,
    tol_p: float = 1e-4, tol_w: float = 1e-4, dq_max: Optional[float] = None,
    w_pos: float = 1.0, w_rot: float = 1.0,
    axis_local: Optional[np.ndarray] = None, axis_target: Optional[np.ndarray] = None,
    aim_length: float = 0.1, tol_axis_deg: float = 1.0, aim_weight: float = 1.0,
) -> Tuple[np.ndarray, IKInfo, np.ndarray]:
    """Right-arm-ONLY (7 DOF), shoulder-frame remote IK -- see
    pin_kinematics.ik_right_arm_dls_pin's own docstring (same function, run
    remotely) for the axis_local/axis_target aim-constraint mode (frees the
    wrist's roll about its own approach axis instead of R_target's full
    3-DOF match; ignores R_target when given). Returns (q7, IKInfo,
    t_shoulder)."""
    payload = {
        "cmd": "ik_right_arm",
        "p_target": np.asarray(p_target, dtype=float).tolist(),
        "R_target": np.asarray(R_target, dtype=float).tolist() if R_target is not None else None,
        "waist_q": list(waist_q),
        "q0_7": np.asarray(q0_7, dtype=float).tolist() if q0_7 is not None else None,
        "joint_bounds": {k: list(v) for k, v in joint_bounds.items()} if joint_bounds else None,
        "waist_weight": waist_weight, "wrist_weight": wrist_weight,
        "max_iters": max_iters, "lamb": lamb, "alpha": alpha,
        "tol_p": tol_p, "tol_w": tol_w, "dq_max": dq_max,
        "w_pos": w_pos, "w_rot": w_rot,
        "axis_local": np.asarray(axis_local, dtype=float).tolist() if axis_local is not None else None,
        "axis_target": np.asarray(axis_target, dtype=float).tolist() if axis_target is not None else None,
        "aim_length": aim_length, "tol_axis_deg": tol_axis_deg, "aim_weight": aim_weight,
    }
    resp = _request(payload)
    q7 = np.array(resp["q7"])
    info = IKInfo(converged=resp["converged"], iters=resp["iters"], final_errors=resp["final_errors"])
    t_shoulder = np.array(resp["t_shoulder"])
    return q7, info, t_shoulder


def gravity_right_arm_remote(q7: np.ndarray, waist_q: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """(7,) generalized gravity torque [Nm], RIGHT_ARM_JOINT_NAMES order -- see
    pin_kinematics.gravity_right_arm_pin's own docstring (same function, run remotely)."""
    resp = _request({"cmd": "gravity_right_arm", "q7": np.asarray(q7, dtype=float).tolist(),
                      "waist_q": list(waist_q)})
    return np.array(resp["g7"])
