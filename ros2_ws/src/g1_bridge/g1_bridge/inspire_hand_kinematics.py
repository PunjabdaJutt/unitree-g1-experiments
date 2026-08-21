"""Shared Inspire RH56DFTP hand kinematics: motor angle -> URDF joint angles.

Each hand has 6 independently actuated motors but g1_29dof_with_inspire_hand.urdf
models 12 joints per hand -- the distal joint of each non-thumb finger, and
the two distal thumb joints, are mechanically coupled to their proximal
joint rather than independently actuated. The URDF encodes that coupling as
<mimic> tags; MuJoCo's URDF compiler in use here does not turn them into
equality constraints (confirmed: model.neq == 0 after loading it), so the
ratios below replicate the URDF's own <mimic> multiplier/parent pairs by
hand. Used by both inspire_hand_modbus_bridge.py (the angle source) and, if
consumers other than robot_state_publisher/mujoco_mirror ever need it,
anything else that wants motor-q -> full joint set.

DOF order shared by every Inspire register group and by unitree_hg's own
MotorState indexing: (0) pinky, (1) ring, (2) middle, (3) index,
(4) thumb_bend, (5) thumb_rotation.
"""

MOTOR_ORDER = ['pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotation']

MOTOR_TO_PRIMARY_JOINT = {
    'pinky': 'little_1_joint',
    'ring': 'ring_1_joint',
    'middle': 'middle_1_joint',
    'index': 'index_1_joint',
    'thumb_bend': 'thumb_2_joint',
    'thumb_rotation': 'thumb_1_joint',
}

# Upper limit (radians) of each primary joint, i.e. its value at q=0 (fully
# closed/flexed). Lower limit is 0 for all of them (q=1, fully open).
PRIMARY_JOINT_UPPER = {
    'little_1_joint': 1.4381,
    'ring_1_joint': 1.4381,
    'middle_1_joint': 1.4381,
    'index_1_joint': 1.4381,
    'thumb_2_joint': 0.5864,
    'thumb_1_joint': 1.1641,
}

# child joint -> (parent joint, multiplier). Order matters: thumb_4_joint
# mimics thumb_3_joint, which must already be computed earlier in this list.
MIMIC_JOINTS = [
    ('index_2_joint', 'index_1_joint', 1.0843),
    ('middle_2_joint', 'middle_1_joint', 1.0843),
    ('ring_2_joint', 'ring_1_joint', 1.0843),
    ('little_2_joint', 'little_1_joint', 1.0843),
    ('thumb_3_joint', 'thumb_2_joint', 0.8024),
    ('thumb_4_joint', 'thumb_3_joint', 0.9487),
]


def expand_hand_angles(side: str, motor_qs: list) -> dict:
    """motor_qs: 6 values in MOTOR_ORDER, each 0 (closed/flexed) - 1 (open/extended).

    Returns {joint_name: angle_radians} for all 12 of that hand's URDF joints.
    """
    angles = {}
    for motor_name, q in zip(MOTOR_ORDER, motor_qs):
        joint = MOTOR_TO_PRIMARY_JOINT[motor_name]
        angles[f'{side}_{joint}'] = PRIMARY_JOINT_UPPER[joint] * (1.0 - q)

    for child_suffix, parent_suffix, multiplier in MIMIC_JOINTS:
        angles[f'{side}_{child_suffix}'] = multiplier * angles[f'{side}_{parent_suffix}']

    return angles
