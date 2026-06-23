"""Casbot DDS Simulator Configuration."""

import os

# Project paths
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
ROBOT_SCENE = os.path.join(PROJECT_ROOT, "casbot_skeleton", "scene.xml")

# Simulation
SIMULATE_DT = 0.005     # 200 Hz physics
VIEWER_DT  = 0.02       #  50 Hz viewer sync

# DDS
DOMAIN_ID = 1
INTERFACE = "lo"        # loopback (same machine as controller)

# Options
USE_JOYSTICK            = True
JOYSTICK_TYPE           = "xbox"
PRINT_SCENE_INFORMATION  = False
ENABLE_ELASTIC_BAND     = False
