"""Casbot DDS Bridge — PD control + DDS pub/sub for 25-DOF Casbot.

Adapted from G1's unitree_sdk2py_bridge.py. Uses unitree_hg IDL
(35 motor slots) — casbot uses first 25 slots.
"""

import mujoco
import numpy as np
import struct
import pygame
import sys

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

TOPIC_LOWCMD  = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3           # pos, vel, torque per motor
NUM_MOTOR        = 25          # casbot 25-DOF
DIM_MOTOR_SENSOR = MOTOR_SENSOR_NUM * NUM_MOTOR  # 75


class CasbotDdsBridge:
    """DDS bridge: subscribes LowCmd → PD control → publishes LowState."""

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.num_motor = self.mj_model.nu        # 25
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.dt = self.mj_model.opt.timestep

        self.joystick = None
        self.have_imu = False
        self.have_frame = False

        # Check available sensors
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":   self.have_imu = True
            if name == "frame_pos":  self.have_frame = True

        # ── Publishers ──
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self._publishLowState, name="sim_lowstate")
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.highStateThread = RecurrentThread(
            interval=self.dt, target=self._publishHighState, name="sim_highstate")
        self.highStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wc_puber = ChannelPublisher(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        self.wc_puber.Init()
        self.wcThread = RecurrentThread(
            interval=0.01, target=self._publishWirelessController, name="sim_wc")
        self.wcThread.Start()

        # ── Subscriber ──
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self._lowCmdHandler, 10)

        print(f"[CasbotBridge] DDS bridge ready. motors={self.num_motor}, dt={self.dt:.4f}, "
              f"imu={'yes' if self.have_imu else 'no'}")

    # ══════════════════════════════════════════════════════════
    #  PD control — called on every LowCmd arrival
    # ══════════════════════════════════════════════════════════

    def _lowCmdHandler(self, msg: LowCmd_):
        """Apply PD: tau = tau_ff + Kp*(q_des - q) + Kd*(dq_des - dq).

        Uses direct MuJoCo API (not sensordata) because casbot XML doesn't
        have explicit jointpos/jointvel sensors. Maps actuator→joint via trnid.
        """
        if self.mj_data is None:
            return
        m = self.mj_model
        d = self.mj_data
        for i in range(self.num_motor):
            jid = m.actuator_trnid[i][0]           # joint id for actuator i
            qadr = m.jnt_qposadr[jid]               # qpos address
            vadr = m.jnt_dofadr[jid]                # qvel address
            q_actual = d.qpos[qadr]
            dq_actual = d.qvel[vadr]
            d.ctrl[i] = (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - q_actual)
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - dq_actual)
            )

    # ══════════════════════════════════════════════════════════
    #  Publish motor state + IMU → LowState
    # ══════════════════════════════════════════════════════════

    def _publishLowState(self):
        if self.mj_data is None:
            return
        m = self.mj_model
        d = self.mj_data
        sd = self.mj_data.sensordata
        nm = self.num_motor

        # Motor states — use direct MuJoCo API (no explicit joint sensors)
        for i in range(nm):
            jid = m.actuator_trnid[i][0]
            qadr = m.jnt_qposadr[jid]
            vadr = m.jnt_dofadr[jid]
            self.low_state.motor_state[i].q       = d.qpos[qadr]
            self.low_state.motor_state[i].dq      = d.qvel[vadr]
            self.low_state.motor_state[i].tau_est = d.actuator_force[i]

        # IMU — from explicit sensors (sensordata[0:10])
        if self.have_imu:
            self.low_state.imu_state.quaternion    = [sd[0], sd[1], sd[2], sd[3]]
            self.low_state.imu_state.gyroscope      = [sd[4], sd[5], sd[6]]
            self.low_state.imu_state.accelerometer  = [sd[7], sd[8], sd[9]]

        # Joystick data
        if self.joystick is not None:
            self._packJoystick()

        self.low_state_puber.Write(self.low_state)

    # ══════════════════════════════════════════════════════════
    #  Publish body state → SportModeState
    # ══════════════════════════════════════════════════════════

    def _publishHighState(self):
        if self.mj_data is None:
            return
        sd = self.mj_data.sensordata
        # IMU is at sensordata[0:10], frame sensors at [10:16]
        if self.have_frame:
            self.high_state.position[0] = sd[10]
            self.high_state.position[1] = sd[11]
            self.high_state.position[2] = sd[12]
            self.high_state.velocity[0] = sd[13]
            self.high_state.velocity[1] = sd[14]
            self.high_state.velocity[2] = sd[15]
            self.high_state_puber.Write(self.high_state)

    # ══════════════════════════════════════════════════════════
    #  Publish joystick → WirelessController
    # ══════════════════════════════════════════════════════════

    def _publishWirelessController(self):
        if self.joystick is None:
            return
        pygame.event.get()
        self.wireless_controller.keys = self._getKeyValue()
        self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
        self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
        self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
        self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])
        self.wc_puber.Write(self.wireless_controller)

    # ══════════════════════════════════════════════════════════
    #  Joystick helpers
    # ══════════════════════════════════════════════════════════

    def _packJoystick(self):
        btns = [0] * 16
        btns[0]  = int(self.joystick.get_button(self.button_id["RB"]))     # R1
        btns[1]  = int(self.joystick.get_button(self.button_id["LB"]))     # L1
        btns[2]  = int(self.joystick.get_button(self.button_id["START"]))   # start
        btns[3]  = int(self.joystick.get_button(self.button_id["SELECT"]))  # select
        btns[4]  = int(self.joystick.get_axis(self.axis_id["RT"]) > 0)     # R2
        btns[5]  = int(self.joystick.get_axis(self.axis_id["LT"]) > 0)     # L2
        btns[8]  = int(self.joystick.get_button(self.button_id["A"]))      # A
        btns[9]  = int(self.joystick.get_button(self.button_id["B"]))      # B
        btns[10] = int(self.joystick.get_button(self.button_id["X"]))      # X
        btns[11] = int(self.joystick.get_button(self.button_id["Y"]))      # Y
        btns[12] = int(self.joystick.get_hat(0)[1] > 0)                    # up
        btns[13] = int(self.joystick.get_hat(0)[0] > 0)                    # right
        btns[14] = int(self.joystick.get_hat(0)[1] < 0)                    # down
        btns[15] = int(self.joystick.get_hat(0)[0] < 0)                    # left

        key_val = sum(btns[i] << i for i in range(16))
        self.low_state.wireless_remote[2] = key_val & 0xFF
        self.low_state.wireless_remote[3] = (key_val >> 8) & 0xFF

        # Joystick axes (LX, RX, RY, LY) packed as float bytes
        for idx, axis_name in enumerate(["LX", "RX", "RY", "LY"]):
            val = self.joystick.get_axis(self.axis_id[axis_name])
            if axis_name in ("LY", "RY"):
                val = -val
            packed = struct.pack("f", val)
            base = 4 + idx * 4
            self.low_state.wireless_remote[base:base + 4] = packed[0:4]

    def _getKeyValue(self):
        state = [0] * 16
        state[0]  = self.joystick.get_button(self.button_id["RB"])
        state[1]  = self.joystick.get_button(self.button_id["LB"])
        state[2]  = self.joystick.get_button(self.button_id["START"])
        state[3]  = self.joystick.get_button(self.button_id["SELECT"])
        state[4]  = int(self.joystick.get_axis(self.axis_id["RT"]) > 0)
        state[5]  = int(self.joystick.get_axis(self.axis_id["LT"]) > 0)
        state[8]  = self.joystick.get_button(self.button_id["A"])
        state[9]  = self.joystick.get_button(self.button_id["B"])
        state[10] = self.joystick.get_button(self.button_id["X"])
        state[11] = self.joystick.get_button(self.button_id["Y"])
        state[12] = int(self.joystick.get_hat(0)[1] > 0)
        state[13] = int(self.joystick.get_hat(0)[0] > 0)
        state[14] = int(self.joystick.get_hat(0)[1] < 0)
        state[15] = int(self.joystick.get_hat(0)[0] < 0)
        return sum(state[i] << i for i in range(16))

    def setupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[CasbotBridge] No gamepad detected.")
            return
        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                "LT": 2, "RT": 5, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 2, "Y": 3, "B": 1, "A": 0,
                "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
            }
        else:  # switch
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        print(f"[CasbotBridge] Joystick connected: {self.joystick.get_name()}")
