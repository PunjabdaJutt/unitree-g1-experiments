#!/home/ammar/miniconda3/envs/pinocchio/bin/python3
# pin_kinematics_server.py
"""Local RPC server exposing pin_kinematics.py's IK/FK to processes that
CAN'T import pinocchio directly -- specifically this repo's own .venv,
where the pip-installed `pin` package is broken (see pin_kinematics.py's
own module docstring for the exact ABI error and why it isn't fixable by
re-pinning versions). Run this in ITS OWN TERMINAL, under the `pinocchio`
conda env:

    conda activate pinocchio
    python3 pin_kinematics_server.py
    # or, if that shell still has .venv/ROS state leaking in (see
    # pin_kinematics.py's docstring on this), invoke the interpreter by
    # full path instead of relying on `conda activate`:
    /home/ammar/miniconda3/envs/pinocchio/bin/python3 pin_kinematics_server.py

Meanwhile, in a SEPARATE terminal, .venv-based code (grasp_ph.py,
dry_run_grasp.py, or anything else) imports pin_kinematics_client.py and
calls ik_two_hands_dls_remote(...) -- same call signature as
kinematics.ik_two_hands_dls, just routed over a local Unix socket to this
process instead of computed in-process. Two terminals, two independent
Python environments (conda `pinocchio` here, .venv there), no shared
dependency to reconcile -- this is the general pattern for any future
piece that needs a conda-only (or otherwise .venv-incompatible) library:
a small stdlib-only (socket+json) server/client pair like this one, not
merging the environments.

Protocol: length-prefixed JSON over a Unix domain socket (SOCK_PATH below)
-- one request per connection (accept -> read exactly one message -> solve
-> write exactly one response -> close), chosen over a persistent
multi-request connection for simplicity/robustness (a client that crashes
mid-request can't wedge the server's connection state). Every message is a
4-byte big-endian length prefix followed by that many bytes of UTF-8 JSON
-- avoids needing to frame on newlines (some future payload could
plausibly contain one) or guess EOF.

Commands:
    {"cmd": "ping"} -> {"ok": true}
    {"cmd": "fk", "frame": "right_wrist_yaw_link", "q17": [...]}
        -> {"T": [[4x4 row-major]]}
    {"cmd": "ik_two_hands_dls", "pL_target": [x,y,z], "pR_target": [x,y,z],
     "RL_target": [[3x3]] | null, "RR_target": [[3x3]] | null,
     "use_waist": bool, "lock_joint_names": [str,...] | null,
     "waist_weight": float, "wrist_weight": float,
     "joint_bounds": {str: [lo, hi]} | null, "q0": [17 floats] | null,
     "max_iters": int, "lamb": float, "alpha": float,
     "tol_p": float, "tol_w": float, "dq_max": float | null,
     "w_pos": float, "w_rot": float}
        -> {"q": [17 floats], "converged": bool, "iters": int,
            "final_errors": {...}}
    Any missing optional key falls back to ik_two_hands_dls_pin's own
    default (same defaults as kinematics.ik_two_hands_dls -- see that
    function's signature).

    Right-arm-only (7 DOF), shoulder-origin commands -- see
    pin_kinematics.py's own "Right-arm-only" section docstring for what the
    shoulder frame is and why:
    {"cmd": "shoulder_origin", "waist_q": [yaw, roll, pitch]}
        -> {"t": [x,y,z]}
    {"cmd": "fk_right_arm", "q7": [...], "waist_q": [...],
     "frame": "right_wrist_yaw_link"}
        -> {"T": [[4x4 row-major]]}   (shoulder frame)
    {"cmd": "ik_right_arm", "p_target": [x,y,z] (shoulder frame),
     "R_target": [[3x3]] | null, "waist_q": [...], "q0_7": [7 floats] | null,
     "joint_bounds": {str: [lo, hi]} | null,
     "waist_weight": float, "wrist_weight": float,
     "max_iters": int, "lamb": float, "alpha": float,
     "tol_p": float, "tol_w": float, "dq_max": float | null,
     "w_pos": float, "w_rot": float,
     "axis_local": [x,y,z] | null, "axis_target": [x,y,z] (shoulder frame) | null,
     "aim_length": float, "tol_axis_deg": float, "aim_weight": float}
        -> {"q7": [7 floats], "converged": bool, "iters": int,
            "final_errors": {...}, "t_shoulder": [x,y,z]}
    axis_local/axis_target (both optional, both-or-neither): when given,
    solves a partial 2-DOF "aim constraint" (see pin_kinematics.
    ik_right_arm_dls_pin's own docstring) instead of R_target's full 3-DOF
    orientation match -- R_target is ignored if axis_target is present.

    {"cmd": "gravity_right_arm", "q7": [...], "waist_q": [...]}
        -> {"g7": [7 floats]}
    RIGHT_ARM_JOINT_NAMES-order generalized gravity torque [Nm] -- see
    pin_kinematics.gravity_right_arm_pin's own docstring.
"""
import json
import os
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_kinematics import (  # noqa: E402
    build_reduced_model, fk_chain_pin, ik_two_hands_dls_pin,
    shoulder_origin_pin, ik_right_arm_dls_pin, fk_right_arm_pin, gravity_right_arm_pin,
)

SOCK_PATH = "/tmp/g1_pin_kinematics.sock"


def _recv_msg(conn: socket.socket) -> dict:
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack(">I", header)
    body = _recv_exact(conn, length)
    return json.loads(body.decode("utf-8"))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed before sending the expected message length")
        buf += chunk
    return buf


def _send_msg(conn: socket.socket, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack(">I", len(body)) + body)


def _handle(req: dict, model) -> dict:
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}

    if cmd == "fk":
        q17 = np.asarray(req["q17"], dtype=float)
        T = fk_chain_pin(model, model.createData(), q17, req["frame"])
        return {"T": T.tolist()}

    if cmd == "ik_two_hands_dls":
        kwargs = {}
        for key in ("use_waist", "waist_weight", "wrist_weight", "max_iters", "lamb", "alpha",
                    "tol_p", "tol_w", "dq_max", "w_pos", "w_rot"):
            if key in req and req[key] is not None:
                kwargs[key] = req[key]
        RL = req.get("RL_target")
        RR = req.get("RR_target")
        q0 = req.get("q0")
        joint_bounds = req.get("joint_bounds")
        q, info = ik_two_hands_dls_pin(
            pL_target=np.asarray(req["pL_target"], dtype=float),
            pR_target=np.asarray(req["pR_target"], dtype=float),
            RL_target=np.asarray(RL, dtype=float) if RL is not None else None,
            RR_target=np.asarray(RR, dtype=float) if RR is not None else None,
            lock_joint_names=req.get("lock_joint_names"),
            joint_bounds={k: tuple(v) for k, v in joint_bounds.items()} if joint_bounds else None,
            q0=np.asarray(q0, dtype=float) if q0 is not None else None,
            model=model,
            **kwargs,
        )
        return {"q": q.tolist(), "converged": info.converged, "iters": info.iters,
                "final_errors": info.final_errors}

    if cmd == "shoulder_origin":
        waist_q = req.get("waist_q", [0.0, 0.0, 0.0])
        t = shoulder_origin_pin(waist_q, model=model)
        return {"t": t.tolist()}

    if cmd == "fk_right_arm":
        q7 = np.asarray(req["q7"], dtype=float)
        waist_q = req.get("waist_q", [0.0, 0.0, 0.0])
        T = fk_right_arm_pin(q7, waist_q=waist_q,
                              frame_name=req.get("frame", "right_wrist_yaw_link"), model=model)
        return {"T": T.tolist()}

    if cmd == "ik_right_arm":
        kwargs = {}
        for key in ("waist_weight", "wrist_weight", "max_iters", "lamb", "alpha",
                    "tol_p", "tol_w", "dq_max", "w_pos", "w_rot",
                    "aim_length", "tol_axis_deg", "aim_weight"):
            if key in req and req[key] is not None:
                kwargs[key] = req[key]
        R_target = req.get("R_target")
        q0_7 = req.get("q0_7")
        joint_bounds = req.get("joint_bounds")
        axis_local = req.get("axis_local")
        axis_target = req.get("axis_target")
        q7, info, t_shoulder = ik_right_arm_dls_pin(
            p_target=np.asarray(req["p_target"], dtype=float),
            R_target=np.asarray(R_target, dtype=float) if R_target is not None else None,
            axis_local=np.asarray(axis_local, dtype=float) if axis_local is not None else None,
            axis_target=np.asarray(axis_target, dtype=float) if axis_target is not None else None,
            waist_q=req.get("waist_q", [0.0, 0.0, 0.0]),
            q0_7=np.asarray(q0_7, dtype=float) if q0_7 is not None else None,
            joint_bounds={k: tuple(v) for k, v in joint_bounds.items()} if joint_bounds else None,
            model=model,
            **kwargs,
        )
        return {"q7": q7.tolist(), "converged": info.converged, "iters": info.iters,
                "final_errors": info.final_errors, "t_shoulder": t_shoulder.tolist()}

    if cmd == "gravity_right_arm":
        q7 = np.asarray(req["q7"], dtype=float)
        waist_q = req.get("waist_q", [0.0, 0.0, 0.0])
        g7 = gravity_right_arm_pin(q7, waist_q=waist_q, model=model)
        return {"g7": g7.tolist()}

    return {"error": f"unknown cmd {cmd!r}"}


def main():
    print("[INFO] building reduced Pinocchio model...")
    model = build_reduced_model()
    print(f"[INFO] model ready ({model.nq} DOF)")

    if os.path.exists(SOCK_PATH):
        os.remove(SOCK_PATH)  # stale socket from a previous, uncleanly-killed run
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    srv.listen(8)
    print(f"[INFO] listening on {SOCK_PATH} -- Ctrl+C to stop")

    try:
        while True:
            conn, _ = srv.accept()
            try:
                req = _recv_msg(conn)
                try:
                    resp = _handle(req, model)
                except Exception as e:  # noqa: BLE001 -- report the error to the client, don't crash the server over one bad request
                    resp = {"error": f"{type(e).__name__}: {e}"}
                _send_msg(conn, resp)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[INFO] stopping.")
    finally:
        srv.close()
        if os.path.exists(SOCK_PATH):
            os.remove(SOCK_PATH)


if __name__ == "__main__":
    main()
