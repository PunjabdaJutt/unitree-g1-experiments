# syntax=docker/dockerfile:1
#
# Dev/runtime image for talking to the Unitree G1 over ROS2 and viewing it
# in RViz. Single-stage on purpose: this is a personal robotics dev
# environment (one machine, GUI tools included), not a fleet deployment
# target -- see .agents/skills/docker-ros2-development for the multi-stage
# pattern if this ever needs to ship headless.
#
# Usage:
#   docker compose build
#   docker compose run --rm g1 ros2 topic list
#   docker compose run --rm g1 ros2 launch g1_bridge bridge.launch.py
# Note: the "library/ros" Docker Hub image doesn't publish a *-desktop tag
# for Jazzy (only *-ros-base and bare *-jazzy) -- osrf/ros mirrors it with
# the GUI variants included.
ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
    libyaml-cpp-dev

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///etc/cyclonedds.xml

WORKDIR /ros2_ws
COPY ros2_ws/src src

# apt lists are kept around (no `rm -rf` yet) since rosdep shells out to
# apt-get install itself -- an empty index here would fail every apt-backed
# rosdep key. Cleaned up once at the end instead.
RUN if [ ! -d /etc/ros/rosdep ] && [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
      rosdep init; \
    fi && \
    rosdep update && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*

COPY docker/cyclonedds.xml.template /etc/cyclonedds.xml.template
COPY docker/entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
