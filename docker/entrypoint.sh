#!/bin/bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Render cyclonedds.xml from the template, pinning discovery to whichever
# NIC is actually wired to the robot (defaults to eno1 -- the interface
# verified on the dev host; override with -e ROBOT_NET_IFACE=... if yours
# is named differently, e.g. after passing a different USB-ethernet adapter
# through to the container).
sed "s/__ROBOT_NET_IFACE__/${ROBOT_NET_IFACE:-eno1}/" \
    /etc/cyclonedds.xml.template > /etc/cyclonedds.xml

exec "$@"
