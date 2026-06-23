import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import threading
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
#from common.joystick import JoyStick, JoystickButton
from pynput import keyboard as pynput_keyboard


class Keyboard:
    """键盘输入类，使用 pynput 全局监听键盘"""
    def __init__(self):
        self.key_states = {}
        self.key_prev_states = {}
        self.key_pressed_events = {}
        self.key_released_events = {}
        self._listener = None
        self._lock = threading.Lock()
        
        # 键名映射
        self.key_map = {
            '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', '0': '0',
            'NUMPAD1': 'num1', 'NUMPAD2': 'num2', 'NUMPAD3': 'num3',
            'NUMPAD4': 'num4', 'NUMPAD5': 'num5', 'NUMPAD6': 'num6',
            'NUMPAD7': 'num7', 'NUMPAD8': 'num8', 'NUMPAD9': 'num9', 'NUMPAD0': 'num0',
            'F1': 'f1', 'F2': 'f2', 'F3': 'f3', 'F4': 'f4', 'F5': 'f5',
            'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e',
            'F': 'f', 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j',
            'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o',
            'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't',
            'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z',
            'SPACE': 'space',
            'ESCAPE': 'esc',
            'ENTER': 'enter',
            'TAB': 'tab',
            'BACKSPACE': 'backspace',
            'LSHIFT': 'shift', 'RSHIFT': 'shift',
            'LCTRL': 'ctrl_l', 'RCTRL': 'ctrl_r',
            'LALT': 'alt_l', 'RALT': 'alt_r',
            'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
        }
        
        # 启动键盘监听
        self._start_listener()
    
    def _start_listener(self):
        """启动键盘监听器"""
        def on_press(key):
            try:
                # 处理小键盘数字 (Key.np0 ~ Key.np9 或 <96>~<105>)
                k_str = str(key)
                if 'np' in k_str or (k_str.startswith('<') and k_str.endswith('>')):
                    # 小键盘数字: <96>=np0, <97>=np1, ... <105>=np9
                    try:
                        num = int(k_str.strip('<>')) - 96
                        if 0 <= num <= 9:
                            k = f'num{num}'
                        else:
                            k = k_str.lower()
                    except:
                        k = k_str.replace('Key.', '').lower()
                elif hasattr(key, 'char') and key.char:
                    # 字符键 (包括Shift+数字产生的!@#$%^&*())
                    k = key.char.lower()
                else:
                    k = str(key).replace('Key.', '').lower()
                
                with self._lock:
                    self.key_states[k] = True
                print(f"[Keyboard] Key pressed: {k}")
            except Exception as e:
                print(f"[Keyboard] Error on press: {e}")
        
        def on_release(key):
            try:
                k_str = str(key)
                if 'np' in k_str or (k_str.startswith('<') and k_str.endswith('>')):
                    try:
                        num = int(k_str.strip('<>')) - 96
                        if 0 <= num <= 9:
                            k = f'num{num}'
                        else:
                            k = k_str.lower()
                    except:
                        k = k_str.replace('Key.', '').lower()
                elif hasattr(key, 'char') and key.char:
                    k = key.char.lower()
                else:
                    k = str(key).replace('Key.', '').lower()
                
                with self._lock:
                    self.key_states[k] = False
                print(f"[Keyboard] Key released: {k}")
            except Exception as e:
                print(f"[Keyboard] Error on release: {e}")
        
        self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
    
    def stop(self):
        """停止键盘监听"""
        if self._listener:
            self._listener.stop()

    def update(self):
        """更新键盘状态（每帧调用）"""
        with self._lock:
            self.key_pressed_events.clear()
            self.key_released_events.clear()
            
            # 处理key_map中的键
            for key_name, key_char in self.key_map.items():
                current = self.key_states.get(key_char, False)
                prev = self.key_prev_states.get(key_char, False)
                
                if current and not prev:
                    self.key_pressed_events[key_name] = True
                if not current and prev:
                    self.key_released_events[key_name] = True
                
                self.key_prev_states[key_char] = current
            
            # 处理所有在key_states中但不在key_map中的键（如符号键!@#$%）
            mapped_chars = set(self.key_map.values())
            for key_char in list(self.key_states.keys()):
                if key_char not in mapped_chars:
                    current = self.key_states.get(key_char, False)
                    prev = self.key_prev_states.get(key_char, False)
                    
                    if current and not prev:
                        self.key_pressed_events[key_char] = True
                    if not current and prev:
                        self.key_released_events[key_char] = True
                    
                    self.key_prev_states[key_char] = current

    def is_key_pressed(self, key):
        """检测按键是否按下"""
        key_char = self.key_map.get(key, key.lower())
        with self._lock:
            return self.key_states.get(key_char, False)

    def is_key_released(self, key):
        """检测按键是否释放（刚松开）"""
        # 直接检查key_released_events，支持原始键名（包括符号键如!@#$）
        key_lower = key.lower()
        # 检查key_map映射后的名称、原始键名（大写）和原始键名（小写）
        mapped_key = self.key_map.get(key, key_lower)
        return (self.key_released_events.get(key, False) or  # 原始键（如 'NUMPAD1', '!'）
                self.key_released_events.get(mapped_key, False) or  # 映射值（如 'num1'）
                self.key_released_events.get(key_lower, False))  # 小写（如 'numpad1'）

    def is_key_just_pressed(self, key):
        """检测按键是否刚按下"""
        return self.key_pressed_events.get(key, False)

    def get_axis_from_keys(self, neg_key, pos_key):
        """从两个按键获取轴值（-1 到 1）"""
        neg = self.is_key_pressed(neg_key)
        pos = self.is_key_pressed(pos_key)
        if neg and pos:
            return 0.0
        elif neg:
            return -1.0
        elif pos:
            return 1.0
        return 0.0


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    #joystick = JoyStick()
    keyboard = Keyboard()
    Running = True
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                # if(joystick.is_button_pressed(JoystickButton.SELECT)):
                #     Running = False

                # joystick.update()
                keyboard.update()
                
                # --- 手柄控制 ---
                # L3: PASSIVE, START: POS_RESET
                # if joystick.is_button_released(JoystickButton.L3):
                #     state_cmd.skill_cmd = FSMCommand.PASSIVE
                # if joystick.is_button_released(JoystickButton.START):
                #     state_cmd.skill_cmd = FSMCommand.POS_RESET
                # # R1 + A/X/Y/B: 技能切换
                # if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.LOCO
                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_1
                # if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_2
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_3
                # if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.L1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_4
                
                # # 手柄摇杆控制速度
                # state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                # state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                # state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                # --- 键盘控制 ---
                # ESC: 退出
                if keyboard.is_key_pressed('ESCAPE'):
                    print("[Keyboard] ESC pressed -> Exit")
                    Running = False
                # P: PASSIVE, SPACE: POS_RESET
                if keyboard.is_key_released('P'):
                    print("[Keyboard] P released -> PASSIVE mode")
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                if keyboard.is_key_released('SPACE'):
                    print("[Keyboard] SPACE released -> POS_RESET mode")
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                # 技能切换：Shift+主键盘数字 或 直接按小键盘数字
                # Shift + 主键盘数字 (检测shift是否按下，检测数字键或其Shift符号的释放事件)
                if keyboard.is_key_pressed('LSHIFT') or keyboard.is_key_pressed('RSHIFT'):
                    if keyboard.is_key_released('1') or keyboard.is_key_released('!'):
                        print("[Keyboard] Shift+1 released -> LOCO mode")
                        state_cmd.skill_cmd = FSMCommand.LOCO
                    if keyboard.is_key_released('2') or keyboard.is_key_released('@'):
                        print("[Keyboard] Shift+2 released -> SKILL_1 (Dance) mode")
                        state_cmd.skill_cmd = FSMCommand.SKILL_1
                    if keyboard.is_key_released('3') or keyboard.is_key_released('#'):
                        print("[Keyboard] Shift+3 released -> SKILL_2 mode")
                        state_cmd.skill_cmd = FSMCommand.SKILL_2
                    if keyboard.is_key_released('4') or keyboard.is_key_released('$'):
                        print("[Keyboard] Shift+4 released -> SKILL_3 mode")
                        state_cmd.skill_cmd = FSMCommand.SKILL_3
                    if keyboard.is_key_released('5') or keyboard.is_key_released('%'):
                        print("[Keyboard] Shift+5 released -> SKILL_4 mode")
                        state_cmd.skill_cmd = FSMCommand.SKILL_4
                # 小键盘数字 (无需Shift)
                if keyboard.is_key_released('NUMPAD1'):
                    print("[Keyboard] Numpad1 released -> LOCO mode")
                    state_cmd.skill_cmd = FSMCommand.LOCO
                if keyboard.is_key_released('NUMPAD2'):
                    print("[Keyboard] Numpad2 released -> SKILL_1 (Dance) mode")
                    state_cmd.skill_cmd = FSMCommand.SKILL_1
                if keyboard.is_key_released('NUMPAD3'):
                    print("[Keyboard] Numpad3 released -> SKILL_2 mode")
                    state_cmd.skill_cmd = FSMCommand.SKILL_2
                if keyboard.is_key_released('NUMPAD4'):
                    print("[Keyboard] Numpad4 released -> SKILL_3 mode")
                    state_cmd.skill_cmd = FSMCommand.SKILL_3
                if keyboard.is_key_released('NUMPAD5'):
                    print("[Keyboard] Numpad5 released -> SKILL_4 mode")
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                
                # Shift + WASD/QE 控制移动 (需要按住Shift)
                if keyboard.is_key_pressed('LSHIFT') or keyboard.is_key_pressed('RSHIFT'):
                    key_vx = keyboard.get_axis_from_keys('S', 'W')  # 前后
                    key_vy = keyboard.get_axis_from_keys('A', 'D')  # 左右
                    key_vyaw = keyboard.get_axis_from_keys('Q', 'E')  # 旋转
                else:
                    key_vx = 0.0
                    key_vy = 0.0
                    key_vyaw = 0.0
                
                # 合并键盘和手柄输入（键盘优先级，有输入则覆盖手柄）
                if key_vx != 0:
                    state_cmd.vel_cmd[0] = key_vx
                if key_vy != 0:
                    state_cmd.vel_cmd[1] = key_vy
                if key_vyaw != 0:
                    state_cmd.vel_cmd[2] = key_vyaw
                
                # 方向键也控制移动
                arrow_vx = keyboard.get_axis_from_keys('DOWN', 'UP')
                arrow_vy = keyboard.get_axis_from_keys('LEFT', 'RIGHT')
                if arrow_vx != 0:
                    state_cmd.vel_cmd[0] = arrow_vx
                if arrow_vy != 0:
                    state_cmd.vel_cmd[1] = -arrow_vy  # 方向键左右反向
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
            except ValueError as e:
                print(str(e))
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        keyboard.stop()
