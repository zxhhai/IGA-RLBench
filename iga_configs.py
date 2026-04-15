from rlbench.tasks import *

class IGAConfig:
    dof_trans: tuple = (True, True, True, False, False, False)
    dof_rot: tuple = (False, False, False, True, True, True)
    num_neg_trans: int = 100
    num_steps_trans: int = 70
    step_size_trans: float = 0.1
    step_size_decay_trans: float = 0.05
    noise_scale_init_trans: float = 0.02
    noise_decay_trans: float = 0.2
    
    num_neg_rot: int = 100
    num_steps_rot: int = 70
    step_size_rot: float = 0.1
    step_size_decay_rot: float = 0.05
    noise_scale_init_rot: float = 0.02
    noise_decay_rot: float = 0.2
    
    # Shared limits and thresholds
    max_trans: float = 0.03
    max_rot: float = 0.1
    fine_align_threshold_trans: float = 0.01
    fine_align_threshold_rot: float = 0.1
    gamma: float = 5.0
    horizon_min: int = 16
    k_min: int = 4
    k_max: int = 12
    ignore_suspicious_rotation_jump: bool = False
    max_steps_per_phase: int = 200

TASK_CONFIGS = {
    # ==========================================
    # 1. 简单触碰类 (Single-Stage: Tool-Like)
    # phase 1: 夹爪 -> 按钮/开关/推块
    # ==========================================
    'push_button': {
        'paradigm': 'single_stage',
        'phase_1_target': ['push_button_target'],
        'restrict_rot': True,
        'gripper_mode': 'constant',
        'gripper_value': 0.0,
    },
    'lamp_on': {
        'paradigm': 'single_stage',
        'phase_1_target': ['lamp_button', 'push_button_target'],
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
        'phase_2_target': ['basket_ball_hoop'],
        'restrict_rot': True,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.08,
    },
    'phone_on_base': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['phone'],
        'phase_2_target': ['phone_case'],
        'restrict_rot': True,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.05,
    },
    'put_rubbish': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['rubbish'],
        'phase_2_target': ['bin'],
        'restrict_rot': True,
        'gripper_mode': 'pick_and_place',
        'dist_threshold': 0.05,
    },
    'plate_out': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['plate'],
        'phase_2_target': ['dish_rack'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'toilet_roll_off': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['toilet_roll'],
        'phase_2_target': ['stand_base', 'holder'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
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
        'ignore_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'open_microwave': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['microwave_door'],
        'phase_2_target': ['microwave'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'close_microwave': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['microwave_door'],
        'phase_2_target': ['microwave'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'toilet_seat_up': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['toilet_seat'],
        'phase_2_target': ['toilet'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'toilet_seat_down': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['toilet_seat'],
        'phase_2_target': ['toilet'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'open_box': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['box_lid'],
        'phase_2_target': ['box_base'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'close_box': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['box_lid'],
        'phase_2_target': ['box_base'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'umbrella_out': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['umbrella'],
        'phase_2_target': ['stand'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'buzz': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['wand'],
        'phase_2_target': ['beat_the_buzz'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
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
}

TASK_NAMES_MAP = {
    'lift_lid': TakeLidOffSaucepan,
    'phone_on_base': PhoneOnBase,
    'open_box': OpenBox,
    'slide_block': SlideBlockToTarget,
    'close_box': CloseBox,
    'basketball': BasketballInHoop,
    'buzz': BeatTheBuzz,
    'close_microwave': CloseMicrowave,
    'plate_out': TakePlateOffColoredDishRack,
    'toilet_seat_down': ToiletSeatDown,
    'toilet_seat_up': ToiletSeatUp,
    'toilet_roll_off': TakeToiletRollOffStand,
    'open_microwave': OpenMicrowave,
    'lamp_on': LampOn,
    'umbrella_out': TakeUmbrellaOutOfUmbrellaStand,
    'push_button': PushButton,
    'put_rubbish': PutRubbishInBin,
}

def get_task_config(task_name):
    """根据任务名称获取配置参数"""
    return TASK_CONFIGS.get(task_name, TASK_CONFIGS['default'])
