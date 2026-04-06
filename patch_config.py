import re

with open('/root/autodl-tmp/run_iga_sim.py', 'r') as f:
    content = f.read()

new_config = """TASK_CONFIGS = {
    # ==========================================
    # 1. 简单触碰类 (Single-Stage: Tool-Like)
    # phase 1: 夹爪 -> 按钮/开关/推块
    # ==========================================
    'push_button': {
        'paradigm': 'single_stage',
        'phase_1_target': ['push_button_target'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 1.0,
    },
    'lamp_on': {
        'paradigm': 'single_stage',
        'phase_1_target': ['lamp_button'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 1.0,
    },
    'slide_block': {
        'paradigm': 'single_stage',
        'phase_1_target': ['block'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 1.0,
    },

    # ==========================================
    # 2. 标准取放类 (Two-Stage: Pick-and-Place)
    # phase 1: 夹爪 -> 待抓取物体
    # phase 2: 手里物体 -> 目标容器/基座
    # ==========================================
    'basketball': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['ball'],
        'phase_2_target': ['basket_hoop'],
        'restrict_rot': False,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.08,
    },
    'phone_on_base': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['phone'],
        'phase_2_target': ['phone_base'],
        'restrict_rot': False,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.05,
    },
    'put_rubbish': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['rubbish'],
        'phase_2_target': ['bin'],
        'restrict_rot': False,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.05,
    },
    'plate_out': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['plate'],
        'phase_2_target': ['dish_rack'],
        'restrict_rot': False,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'toilet_roll_off': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['toilet_roll'],
        'phase_2_target': ['toilet_roll_stand'],
        'restrict_rot': False,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },

    # ==========================================
    # 3. 约束关节类 (Two-Stage: Articulated / Tool-Use)
    # phase 1: 夹爪 -> 把手/边缘
    # phase 2: 把手/边缘 -> 目标开启状态/支架
    # ==========================================
    'lift_lid': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['saucepan_lid'],
        'phase_2_target': ['saucepan'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'open_microwave': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['microwave_door'],
        'phase_2_target': ['microwave'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'close_microwave': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['microwave_door'],
        'phase_2_target': ['microwave'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'toilet_seat_up': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['toilet_seat'],
        'phase_2_target': ['toilet'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'toilet_seat_down': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['toilet_seat'],
        'phase_2_target': ['toilet'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'open_box': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['box_lid'],
        'phase_2_target': ['box'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'close_box': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['box_lid'],
        'phase_2_target': ['box'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'umbrella_out': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['umbrella'],
        'phase_2_target': ['umbrella_stand'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'buzz': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['stubby_handle'],
        'phase_2_target': ['buzz_wire'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },

    # --- 默认配置 ---
    'default': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['item'],
        'phase_2_target': ['target'],
        'restrict_rot': False,
        'gripper_mode': 'proximity_grasp',
        'dist_threshold': 0.05,
    }
}"""

pattern = re.compile(r'TASK_CONFIGS = \{.*?\n\}(?=\n\nTASK_NAMES_MAP = \{)', re.DOTALL)
new_content = pattern.sub(new_config, content)

with open('/root/autodl-tmp/run_iga_sim.py', 'w') as f:
    f.write(new_content)

print(f"Patched properly? {content != new_content}")
