# g1_depth_msg.py
"""Custom DDS/IDL message type for streaming the G1's D435i depth camera
(color + aligned depth) over raw CycloneDDS via unitree_sdk2py's
ChannelPublisher/ChannelSubscriber -- the same primitives camera_grabber.py
already uses for the Mid-360 lidar (rt/utlidar/cloud_livox_mid360), and the
confirmed-working alternative (per this repo's own prior finding) to
subscribing over ROS2's rmw layer, which is specifically what fails for
Unitree's own /frontvideostream. This topic is NOT a Unitree firmware
topic -- it's this repo's own, published from depth_relay_onboard.py
running on the onboard Jetson.

MUST be the byte-for-byte identical file on both the publisher (onboard
Jetson) and every subscriber (workstation) -- CycloneDDS matches
readers/writers by type signature, so copy this file rather than
maintaining separate versions. Defined the same way Unitree's own vendored
IDL types are (see
.unitree_sdk2_python_src/unitree_sdk2py/idl/unitree_go/msg/dds_/_Go2FrontVideoData_.py)
-- a plain dataclass decorated with cyclonedds.idl, no idlc compile step
needed.
"""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

TOPIC_NAME = "g1_depth_camera/color_depth_frame"


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthColorFrame_(idl.IdlStruct, typename="g1_depth_camera.msg.dds_.DepthColorFrame_"):
    time_frame: types.uint64
    width: types.uint32
    height: types.uint32
    fx: types.float32
    fy: types.float32
    cx: types.float32
    cy: types.float32
    depth_scale: types.float32
    color_jpeg: types.sequence[types.uint8]  # JPEG-encoded color_bgr
    depth_png: types.sequence[types.uint8]   # PNG-encoded uint16 depth, aligned to color
