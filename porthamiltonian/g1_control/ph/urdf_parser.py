"""URDF parser for the G1 PH layer.

Produces LinkInertial / JointSpec / URDFModel dataclasses. Stdlib only
(xml.etree). No XACRO expansion needed for the G1 DFQ URDF.

Joint-index contract (see memory/ph_g1_design_map.md):
    URDF joint order here is the parse order. Alignment with IsaacLab's
    articulation order (robot.data.joint_names) is done at runtime via
    name-based lookup using `idx_of_joint` (mirrors lift_fingerTorque.py:501).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class LinkInertial:
    name: str
    mass: float
    com: np.ndarray              # (3,) COM in link frame
    inertia: np.ndarray          # (3,3) inertia tensor about COM in link frame
    has_inertial: bool = True    # False if URDF omitted <inertial>


@dataclass
class JointSpec:
    name: str
    type: str                    # "revolute" | "prismatic" | "fixed" | "continuous"
    parent: str
    child: str
    origin_xyz: np.ndarray       # (3,) translation parent->joint
    origin_rpy: np.ndarray       # (3,) RPY (Rz*Ry*Rx convention)
    axis: np.ndarray             # (3,) unit axis in joint frame (zeros for fixed)
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    mimic_joint: Optional[str] = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class URDFModel:
    name: str
    links: Dict[str, LinkInertial] = field(default_factory=dict)
    joints: List[JointSpec] = field(default_factory=list)   # parse order
    joints_by_name: Dict[str, JointSpec] = field(default_factory=dict)
    root_link: str = ""
    children_of: Dict[str, List[str]] = field(default_factory=dict)

    # -- convenience -----------------------------------------------------------
    def movable_joints(self) -> List[JointSpec]:
        return [j for j in self.joints if j.type not in ("fixed",)]

    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    def movable_joint_names(self) -> List[str]:
        return [j.name for j in self.movable_joints()]


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def _vec3(s: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


def _parse_inertial(link_el: ET.Element, name: str) -> LinkInertial:
    inertial = link_el.find("inertial")
    if inertial is None:
        return LinkInertial(
            name=name, mass=0.0,
            com=np.zeros(3), inertia=np.zeros((3, 3)),
            has_inertial=False,
        )
    origin = inertial.find("origin")
    com = _vec3(origin.get("xyz") if origin is not None else None)
    mass_el = inertial.find("mass")
    mass = float(mass_el.get("value")) if mass_el is not None else 0.0
    I = inertial.find("inertia")
    ixx = float(I.get("ixx", 0.0)); iyy = float(I.get("iyy", 0.0)); izz = float(I.get("izz", 0.0))
    ixy = float(I.get("ixy", 0.0)); ixz = float(I.get("ixz", 0.0)); iyz = float(I.get("iyz", 0.0))
    inertia = np.array([[ixx, ixy, ixz],
                        [ixy, iyy, iyz],
                        [ixz, iyz, izz]], dtype=float)
    return LinkInertial(name=name, mass=mass, com=com, inertia=inertia)


def _parse_joint(j_el: ET.Element) -> JointSpec:
    name = j_el.get("name"); jtype = j_el.get("type")
    origin = j_el.find("origin")
    xyz = _vec3(origin.get("xyz") if origin is not None else None)
    rpy = _vec3(origin.get("rpy") if origin is not None else None)
    axis_el = j_el.find("axis")
    axis = _vec3(axis_el.get("xyz") if axis_el is not None else None,
                 default=(0.0, 0.0, 0.0))
    parent = j_el.find("parent").get("link")
    child = j_el.find("child").get("link")
    lim = j_el.find("limit")
    lower = float(lim.get("lower", 0.0)) if lim is not None else 0.0
    upper = float(lim.get("upper", 0.0)) if lim is not None else 0.0
    effort = float(lim.get("effort", 0.0)) if lim is not None else 0.0
    velocity = float(lim.get("velocity", 0.0)) if lim is not None else 0.0
    mimic_el = j_el.find("mimic")
    mimic_joint = mimic_el.get("joint") if mimic_el is not None else None
    mimic_mul = float(mimic_el.get("multiplier", 1.0)) if mimic_el is not None else 1.0
    mimic_off = float(mimic_el.get("offset", 0.0)) if mimic_el is not None else 0.0
    return JointSpec(
        name=name, type=jtype, parent=parent, child=child,
        origin_xyz=xyz, origin_rpy=rpy, axis=axis,
        lower=lower, upper=upper, effort=effort, velocity=velocity,
        mimic_joint=mimic_joint, mimic_multiplier=mimic_mul, mimic_offset=mimic_off,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def parse_urdf(path: str | Path) -> URDFModel:
    tree = ET.parse(str(path))
    root = tree.getroot()
    model = URDFModel(name=root.get("name", ""))

    for link_el in root.findall("link"):
        lname = link_el.get("name")
        model.links[lname] = _parse_inertial(link_el, lname)

    for j_el in root.findall("joint"):
        j = _parse_joint(j_el)
        model.joints.append(j)
        model.joints_by_name[j.name] = j
        model.children_of.setdefault(j.parent, []).append(j.child)

    # Root = link that is never a child.
    children = {j.child for j in model.joints}
    roots = [n for n in model.links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root link, got {roots}")
    model.root_link = roots[0]
    return model


# -----------------------------------------------------------------------------
# Joint alignment helpers (URDF <-> IsaacLab articulation)
# -----------------------------------------------------------------------------

def align_to_articulation(
    urdf_joints: List[str],
    articulation_joint_names: List[str],
) -> Dict[str, int]:
    """Map URDF movable-joint name -> index in articulation order.

    Mirrors the `idx_of_joint` pattern from lift_fingerTorque.py:501.
    Joints present in URDF but not in articulation (e.g. fixed joints removed
    by USD import) are skipped.
    """
    return {
        n: articulation_joint_names.index(n)
        for n in urdf_joints
        if n in articulation_joint_names
    }
