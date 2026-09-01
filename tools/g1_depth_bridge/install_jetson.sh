#!/bin/bash
# install_jetson.sh -- one-time setup for the G1 depth-camera DDS relay.
# Run ON THE ONBOARD JETSON, from inside this directory, after copying the
# whole g1_depth_bridge/ folder there (see README.md).
#
# This does NOT build librealsense/pyrealsense2 -- that's a separate,
# already-completed step on this specific box (RSUSB backend, no kernel
# patch, see README.md's Prerequisites section). This script only installs
# unitree_sdk2py, which the relay needs for DDS publishing and which is
# confirmed NOT already present on this machine (checked 2026-08-31:
# `python3 -c "import unitree_sdk2py"` -> ModuleNotFoundError).
#
# Targeted install only -- does not touch any already-installed package,
# consistent with not running full system upgrades on this box.
set -euo pipefail

if python3 -c "import unitree_sdk2py" 2>/dev/null; then
    echo "[INFO] unitree_sdk2py already importable, skipping install."
else
    # The `cyclonedds` Python package's build step needs the native CycloneDDS
    # C library (headers + .so) findable via CYCLONEDDS_HOME -- it does not
    # build/vendor one itself. Reuse the one Unitree already built here at the
    # matching version (0.10.2) rather than building a second copy.
    CYCLONEDDS_WS_INSTALL="$HOME/cyclonedds_ws/install/cyclonedds"
    if [ ! -d "$CYCLONEDDS_WS_INSTALL" ]; then
        echo "[ERROR] expected native CycloneDDS install not found at $CYCLONEDDS_WS_INSTALL"
        echo "        (this is Unitree's own pre-built ~/cyclonedds_ws on this box --"
        echo "        if it's genuinely missing/moved, the cyclonedds Python package"
        echo "        needs a CYCLONEDDS_HOME pointed at some other CycloneDDS 0.10.2"
        echo "        install with headers, or a fresh from-source build.)"
        exit 1
    fi
    export CYCLONEDDS_HOME="$CYCLONEDDS_WS_INSTALL"
    echo "[INFO] CYCLONEDDS_HOME=$CYCLONEDDS_HOME"

    echo "[INFO] installing unitree_sdk2py (pulls in cyclonedds==0.10.2, numpy, opencv-python)..."
    TMP_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git "$TMP_DIR/unitree_sdk2_python"
    pip3 install "$TMP_DIR/unitree_sdk2_python"
    rm -rf "$TMP_DIR"
    echo "[INFO] unitree_sdk2py installed."
fi

if [ ! -f /usr/local/lib/librealsense2.so.2.58 ]; then
    echo "[WARN] /usr/local/lib/librealsense2.so.2.58 not found -- the from-source"
    echo "       pyrealsense2/librealsense build (RSUSB backend) is a separate"
    echo "       prerequisite this script does not perform. See README.md."
fi

echo "[INFO] setup check complete. Test with: python3 depth_relay_onboard.py"
