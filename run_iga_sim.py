import os
import sys
from pathlib import Path

### fix environment
def _bootstrap_coppeliasim_runtime():
    if os.environ.get('COPPELIASIM_RUNTIME_BOOTSTRAPPED') == '1':
        return

    if sys.argv[0] in {'-c', '-m', '-'}:
        return

    coppeliasim_root = Path('/root/CoppeliaSim')
    qt_lib_dir = coppeliasim_root / 'qt_libs_bak'
    if not coppeliasim_root.exists() or not qt_lib_dir.exists():
        return

    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    ld_paths = [str(qt_lib_dir), str(coppeliasim_root)]
    if ld_library_path:
        ld_paths.append(ld_library_path)

    preload_libs = [
        coppeliasim_root / 'libicudata.so.56.1',
        coppeliasim_root / 'libicuuc.so.56.1',
        coppeliasim_root / 'libicui18n.so.56.1',
        coppeliasim_root / 'liblua5.1.so',
        qt_lib_dir / 'libQt5Core.so.5.12.5',
        qt_lib_dir / 'libQt5Gui.so.5.12.5',
        qt_lib_dir / 'libQt5Widgets.so.5.12.5',
        qt_lib_dir / 'libQt5Network.so.5.12.5',
        qt_lib_dir / 'libQt5OpenGL.so.5.12.5',
        qt_lib_dir / 'libQt5SerialPort.so.5.12.5',
    ]

    preload_values = [str(path) for path in preload_libs if path.exists()]
    existing_preload = os.environ.get('LD_PRELOAD', '')
    if existing_preload:
        preload_values.append(existing_preload)

    new_env = os.environ.copy()
    new_env['COPPELIASIM_RUNTIME_BOOTSTRAPPED'] = '1'
    new_env['COPPELIASIM_ROOT'] = str(coppeliasim_root)
    new_env['QT_PLUGIN_PATH'] = str(coppeliasim_root)
    new_env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(coppeliasim_root / 'platforms')
    new_env['QT_QPA_PLATFORM'] = 'xcb'
    new_env['LD_LIBRARY_PATH'] = ':'.join(ld_paths)
    if preload_values:
        new_env['LD_PRELOAD'] = ' '.join(preload_values)

    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

_bootstrap_coppeliasim_runtime()
###

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iga'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instant_policy'))

import pickle
import numpy as np
import torch
import queue
import threading
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from rlbench.tasks import *
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from sim_utils import get_rgb_montage_frame, save_video_from_frames
from utils import pose_to_transform, transform_to_pose, subsample_pcd

from iga.models.inference_model import IGA
from iga.utils.parser_utils import get_inference_parser
from iga.utils.common_utils import transform_pcd


# ==========================================
# 1. 超参数统一管理 (Hyperparameter Configs)
# ==========================================

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
    fine_align_threshold: float = 0.02
    gamma: float = 15.0
    horizon_min: int = 16
    k_min: int = 4
    k_max: int = 12
    ignore_suspicious_rotation_jump: bool = False

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
        'gripper_mode': 'proximity_grasp',
    },
    'toilet_roll_off': {
        'paradigm': 'two_stage_pick_and_place',
        'phase_1_target': ['toilet_roll'],
        'phase_2_target': ['toilet_roll_stand'],
        'restrict_rot': False,
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
        'phase_2_target': ['box'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'close_box': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['box_lid'],
        'phase_2_target': ['box'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'umbrella_out': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['umbrella'],
        'phase_2_target': ['umbrella_stand'],
        'restrict_rot': True,
        'gripper_mode': 'proximity_grasp',
    },
    'buzz': {
        'paradigm': 'two_stage_articulated',
        'phase_1_target': ['stubby_handle'],
        'phase_2_target': ['buzz_wire'],
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


# ==========================================
# 2. 函数封装，保持主要逻辑的简洁
# ==========================================

def get_task_config(task_name):
    """根据任务名称获取配置参数"""
    return TASK_CONFIGS.get(task_name, TASK_CONFIGS['default'])

def create_sim_env(task_name, headless=False, restrict_rot=True):
    """创建仿真环境"""
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, './', obs_config=obs_config, headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES_MAP[task_name])

    # HACK: Fix get_path issue manually
    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)
    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([
        6.74760377e-05, -1.91104114e-02, -3.62065766e-05, 
        -1.64271665e+00, -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    if restrict_rot:
        rot_bounds = env._scene.task.base_rotation_bounds()
        mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
        env._scene.task.base_rotation_bounds = lambda: (
            (0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))
    
    return env, task

def get_demos_with_timeout(task, num_demos, live_demos=True, max_attempts=1, timeout_sec=90):
    """带超时机制的Demo获取函数"""
    result_queue = queue.Queue(maxsize=1)
    def _worker():
        try:
            demos = task.get_demos(num_demos, live_demos=live_demos, max_attempts=max_attempts)
            result_queue.put((True, demos))
        except Exception as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)
    if worker.is_alive():
        raise TimeoutError(f"task.get_demos timed out after {timeout_sec}s")
    ok, payload = result_queue.get_nowait()
    if ok: return payload
    raise payload

def extract_pcds_from_obs(obs, mask_ids=None):
    """从观察中提取点云数据"""
    pcds = []
    for cam in ('front', 'left_shoulder', 'right_shoulder'):
        cam_pcd = getattr(obs, f'{cam}_point_cloud')
        cam_mask = getattr(obs, f'{cam}_mask')
        mask = np.isin(cam_mask, mask_ids)
        pcds.append(cam_pcd[mask])
    
    pcd = np.concatenate(pcds, axis=0) if pcds else np.zeros((0,3))
    
    if pcd.shape[0] < 20:
        return None
        
    pcd = subsample_pcd(pcd, 1024)
    return pcd

def get_object_ids(task, name_list):
    """根据名字关键字获取环境中的物体ID"""
    all_objs = task._scene.pyrep.get_objects_in_tree()
    ids = []
    for o in all_objs:
        name = o.get_name().lower()
        if any(keyword in name for keyword in name_list):
            ids.append(o.get_handle())
    return ids

def split_transform_to_horizon(T_final, horizon=5):
    """将一个 T_final 变换拆解为 horizon 步的小变换序列"""
    final_trans = T_final[:3, 3]
    final_rot_matrix = T_final[:3, :3]
    final_quat = Rot.from_matrix(final_rot_matrix).as_quat()
    
    start_trans = np.zeros(3)
    start_quat = np.array([0, 0, 0, 1])
    
    actions_list = []
    key_times = [0, 1]
    key_rots = Rot.from_quat([start_quat, final_quat])
    slerp = Slerp(key_times, key_rots)
    
    for i in range(1, horizon + 1):
        alpha = i / horizon
        interp_trans = (1 - alpha) * start_trans + alpha * final_trans
        interp_rot = slerp(alpha).as_matrix()
        
        T_step = np.eye(4)
        T_step[:3, :3] = interp_rot
        T_step[:3, 3] = interp_trans
        actions_list.append(T_step)
        
    return actions_list

def extract_demo_features(demos, gripper_pcd_ee, task, task_cfg, save_video=False, video_dir=None, task_name="task"):
    """根据两阶段对齐范式提取演示特征"""
    demo_pcds_phase_1 = {'pcds_grasped': [], 'pcds_target': []}
    demo_pcds_phase_2 = {'pcds_grasped': [], 'pcds_target': []}
    
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = get_object_ids(task, task_cfg.get('phase_1_target', []))
    phase_2_target_ids = get_object_ids(task, task_cfg.get('phase_2_target', []))
    
    # Store IDs back in config so rollout can reuse them
    task_cfg['phase_1_target_ids'] = phase_1_target_ids
    task_cfg['phase_2_target_ids'] = phase_2_target_ids

    for i, d in enumerate(demos):
        if save_video and video_dir:
            demo_frames = [get_rgb_montage_frame(o) for o in d if get_rgb_montage_frame(o) is not None]
            if demo_frames:
                demo_video_path = Path(video_dir) / f'iga_{task_name}_demo_{i}.mp4'
                save_video_from_frames(demo_frames, demo_video_path, fps=15)
        
        # 寻找夹爪首次闭合的帧 (或者是抓到物体的帧)
        grasp_idx = len(d) - 1
        for j, o in enumerate(d):
            if o.gripper_open < 0.5:
                grasp_idx = j
                break
                
        # Phase 1: Reach / Grasp
        if paradigm in ['two_stage_pick_and_place', 'two_stage_articulated', 'single_stage']:
            grasp_obs = d[grasp_idx]
            grasp_T_w_e = pose_to_transform(grasp_obs.gripper_pose)
            
            p1_target_pcd = extract_pcds_from_obs(grasp_obs, mask_ids=phase_1_target_ids)
            p1_source_pcd = transform_pcd(gripper_pcd_ee, grasp_T_w_e)
            
            if p1_target_pcd is not None:
                inv_grasp_T_w_e = np.linalg.inv(grasp_T_w_e)
                p1_source_local = transform_pcd(p1_source_pcd, inv_grasp_T_w_e)
                p1_target_local = transform_pcd(p1_target_pcd, inv_grasp_T_w_e)
                demo_pcds_phase_1['pcds_grasped'].append(p1_source_local)
                demo_pcds_phase_1['pcds_target'].append(p1_target_local)

        # Phase 2: Manipulate / Place
        if paradigm in ['two_stage_pick_and_place', 'two_stage_articulated']:
            final_obs = d[-1]
            final_T_w_e = pose_to_transform(final_obs.gripper_pose)
            
            p2_target_pcd = extract_pcds_from_obs(final_obs, mask_ids=phase_2_target_ids)
            p2_source_pcd = extract_pcds_from_obs(final_obs, mask_ids=phase_1_target_ids)
            
            if p2_target_pcd is not None and p2_source_pcd is not None:
                inv_final_T_w_e = np.linalg.inv(final_T_w_e)
                p2_source_local = transform_pcd(p2_source_pcd, inv_final_T_w_e)
                p2_target_local = transform_pcd(p2_target_pcd, inv_final_T_w_e)
                demo_pcds_phase_2['pcds_grasped'].append(p2_source_local)
                demo_pcds_phase_2['pcds_target'].append(p2_target_local)
                
    return demo_pcds_phase_1, demo_pcds_phase_2

# ==========================================
# 3. 对一组demos执行多次rollout测试
# ==========================================

def get_gripper_action(config, dist_to_obj, dist_to_target, is_grasped):
    mode = config.get('gripper_mode', 'constant')
    
    if mode == 'constant':
        return config.get('gripper_value', 1.0)
        
    if mode == 'proximity_grasp':
        # 如果还没抓到且距离够近，或者已经抓到了，就闭合
        if dist_to_obj < 0.03 or is_grasped:
            return 0.0
        return 1.0
        
    if mode == 'pick_and_place':
        if not is_grasped:
            return 0.0 if dist_to_obj < 0.03 else 1.0
        else:
            # 已经抓到了，判断是否到达终点该放手了
            return 1.0 if dist_to_target < config.get('dist_threshold', 0.05) else 0.0
            
    return 1.0

def run_rollout(iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, task_cfg, save_video=False, rollout_idx=0, cfg=None):
    """两阶段单次rollout测试"""
    if cfg is None: cfg = IGAConfig()
        
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = task_cfg.get('phase_1_target_ids', [])
    phase_2_target_ids = task_cfg.get('phase_2_target_ids', [])
    
    descriptions, obs = task.reset()
    max_steps_per_phase = 30
    rollout_frames = []
    success = False
    
    if save_video:
        f = get_rgb_montage_frame(obs)
        if f is not None: rollout_frames.append(f)
            
    def run_phase(phase, max_steps):
        nonlocal obs, success, rollout_frames
        
        last_valid_target_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
        if last_valid_target_pcd is None: last_valid_target_pcd = np.zeros((1024, 3))
        
        source_pcds_demo = demo_pcds_p1 if phase == 1 else demo_pcds_p2
        last_valid_source_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids) if phase == 2 else None
        if phase == 2 and last_valid_source_pcd is None: last_valid_source_pcd = np.zeros((1024, 3))
            
        p2_center = np.mean(last_valid_target_pcd, axis=0) if last_valid_target_pcd.shape[0] > 0 else None

        for step in range(max_steps):
            current_T_w_e = pose_to_transform(obs.gripper_pose)
            
            # 1. Update Target
            current_target_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
            if current_target_pcd is not None:
                last_valid_target_pcd = current_target_pcd
            else:
                current_target_pcd = last_valid_target_pcd
                
            # 2. Update Source
            if phase == 1:
                source_pcd = transform_pcd(gripper_pcd_ee, current_T_w_e)
            else:
                source_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids)
                if source_pcd is not None:
                    last_valid_source_pcd = source_pcd
                else:
                    source_pcd = last_valid_source_pcd
                    
            inv_current_T_w_e = np.linalg.inv(current_T_w_e)
            source_pcd_local = transform_pcd(source_pcd, inv_current_T_w_e)
            current_target_pcd_local = transform_pcd(current_target_pcd, inv_current_T_w_e)
                
            live_pcds = {'pcd_grasped': source_pcd_local, 'pcd_target': current_target_pcd_local}
            
            T_e_e_new = iga_model.get_transform(source_pcds_demo, live_pcds, visualise=False)
            
            trans_dist_total = np.linalg.norm(T_e_e_new[:3, 3])
            rot_dist_total = np.linalg.norm(Rot.from_matrix(T_e_e_new[:3, :3]).as_rotvec())

            if task_cfg.get('restrict_rot', False) and cfg.ignore_suspicious_rotation_jump and trans_dist_total < cfg.fine_align_threshold and rot_dist_total > 1.57:
                T_e_e_new[:3, :3] = np.eye(3)

            horizon = max(int(trans_dist_total / cfg.max_trans), int(rot_dist_total / cfg.max_rot), cfg.horizon_min)
            execute_steps = min(int(np.round(cfg.k_min + (cfg.k_max - cfg.k_min) * (1 - np.exp(-cfg.gamma * trans_dist_total)))), int(np.round(cfg.k_min + (cfg.k_max - cfg.k_min) * (1 - np.exp(-cfg.gamma * rot_dist_total)))))
            
            actions_list = split_transform_to_horizon(T_e_e_new, horizon=horizon)
            original_T_w_e = current_T_w_e
            
            is_grasped = obs.gripper_open < 0.5
            
            # Extract distance to obj and dist to target
            p1_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids)
            p1_center = np.mean(p1_pcd, axis=0) if p1_pcd is not None and p1_pcd.shape[0] > 0 else np.array([float('inf'), float('inf'), float('inf')])
            dist_to_obj = np.linalg.norm(current_T_w_e[:3, 3] - p1_center)

            p2_pcd = extract_pcds_from_obs(obs, mask_ids=phase_2_target_ids)
            p2_center_active = np.mean(p2_pcd, axis=0) if p2_pcd is not None and p2_pcd.shape[0] > 0 else np.array([float('inf'), float('inf'), float('inf')])
            dist_to_target = np.linalg.norm(current_T_w_e[:3, 3] - p2_center_active)
            
            gripper_action = get_gripper_action(task_cfg, dist_to_obj, dist_to_target, is_grasped)
            
            for T_e_e_interp in actions_list[:execute_steps]:
                target_T_w_e_desired = original_T_w_e @ T_e_e_interp 
                current_T_w_e_actual = pose_to_transform(obs.gripper_pose)
                
                T_delta = np.linalg.inv(current_T_w_e_actual) @ target_T_w_e_desired
                
                trans_diff = T_delta[:3, 3]
                trans_dist = np.linalg.norm(trans_diff)
                if trans_dist > cfg.max_trans:
                    trans_diff = (trans_diff / trans_dist) * cfg.max_trans
                    
                rot_diff = Rot.from_matrix(T_delta[:3, :3]).as_rotvec()
                rot_dist = np.linalg.norm(rot_diff)
                if rot_dist > cfg.max_rot:
                    rot_diff = (rot_diff / rot_dist) * cfg.max_rot
                    
                T_delta_clipped = np.eye(4)
                T_delta_clipped[:3, :3] = Rot.from_rotvec(rot_diff).as_matrix()
                T_delta_clipped[:3, 3] = trans_diff
                
                target_T_w_e = current_T_w_e_actual @ T_delta_clipped
                target_pose_7d = transform_to_pose(target_T_w_e)
                
                action = np.append(target_pose_7d, gripper_action)
                
                try:
                    obs, reward, terminate = task.step(action)
                    if save_video:
                        f = get_rgb_montage_frame(obs)
                        if f is not None: rollout_frames.append(f)
                    if terminate:
                        success = True
                        return True
                except Exception as e:
                    print(f"Task step failed: {e}")
                    return False
                    
            if trans_dist_total < cfg.fine_align_threshold and rot_dist_total < 0.05:
                break
                
        return False
        
    # Execution Flow
    if paradigm == 'single_stage':
        run_phase(1, max_steps_per_phase)
    elif paradigm in ['two_stage_pick_and_place', 'two_stage_articulated']:
        # Phase 1: Reach
        run_phase(1, max_steps_per_phase)
        if success: return True, rollout_frames
        
        # Grasp Action Transition (根据当前模式自适应)
        gripper_action_idx = get_gripper_action(task_cfg, dist_to_obj=0.0, dist_to_target=float('inf'), is_grasped=True)
        action_pose = transform_to_pose(pose_to_transform(obs.gripper_pose))
        action = np.append(action_pose, gripper_action_idx)
        terminate = False
        try:
            for _ in range(5):
                task._scene.step()
            obs, reward, terminate = task.step(action)
        except Exception:
            pass
            
        if save_video:
            f = get_rgb_montage_frame(obs)
            if f is not None: rollout_frames.append(f)
            
        if terminate:
            success = True
            return True, rollout_frames
            
        # Phase 2: Manipulate/Place
        run_phase(2, max_steps_per_phase)
        
    return success, rollout_frames

def evaluate_iga_policy(task_name, iga_model, num_demos=5, num_rollouts=10, save_video=True, video_dir='./videos'):
    """使用给定的Demo，测试多次rollout以计算成功率"""
    video_dir = str(Path(video_dir) / task_name)
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        
    task_cfg = get_task_config(task_name)
    cfg = IGAConfig()
    env, task = create_sim_env(task_name, headless=True, restrict_rot=task_cfg['restrict_rot'])
    
    gripper_pcd_ee = pickle.load(open('./iga/iga/assets/franka_gripper_pcd.pkl', 'rb'))
    gripper_pcd_ee = subsample_pcd(gripper_pcd_ee, 1024)
    
    print(f"Fetcing {num_demos} offline demo(s)...")
    # demos = get_demos_with_timeout(task, num_demos, live_demos=True, max_attempts=10, timeout_sec=120)
    demos = task.get_demos(num_demos, live_demos=True, max_attempts=50)
    
    demo_pcds_p1, demo_pcds_p2 = extract_demo_features(demos, gripper_pcd_ee, task, task_cfg, save_video, video_dir, task_name)
    
    success_count = 0
    print(f"Starting {num_rollouts} evaluation rollouts for task: {task_name}")
    
    for i in range(num_rollouts):
        print(f"--- Rollout {i+1}/{num_rollouts} ---")
        success, frames = run_rollout(iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, task_cfg, save_video, rollout_idx=i, cfg=cfg)
        
        if success:
            success_count += 1
            print(f"Rollout {i+1} Succeded!")
        else:
            print(f"Rollout {i+1} Failed.")
            
        if save_video and frames:
            os.makedirs(video_dir, exist_ok=True)
            res_str = "SUCCESS" if success else "FAIL"
            save_video_from_frames(frames, Path(video_dir) / f'run_{i+1}_{res_str}.mp4', fps=15)
            
    accuracy = success_count / num_rollouts
    print(f"Evaluation Completed! Task: {task_name} | Accuracy: {accuracy*100:.2f}% ({success_count}/{num_rollouts})")
    
    env.shutdown()
    return accuracy


if __name__ == '__main__':
    parser = get_inference_parser()
    # Add task argument directly here so it works even if not defined in the imported parser
    parser.add_argument('--task', type=str, default='push_button', help='Name of the RLBench task to run')
    args, unknown = parser.parse_known_args()
    
    # 获取任务名称以自适应调整参数
    task_name = args.task
    task_cfg = get_task_config(task_name)
    cfg = IGAConfig()

    dof_trans = cfg.dof_trans
    if args.no_x or args.no_y or args.no_z:
        dof_trans = (not args.no_x, not args.no_y, not args.no_z, False, False, False)
        
    dof_rot = cfg.dof_rot
    if args.no_rot_x or args.no_rot_y or args.no_rot_z:
        dof_rot = (False, False, False, not args.no_rot_x, not args.no_rot_y, not args.no_rot_z)

    model_dir = './iga/iga/checkpoints'
    
    iga = IGA(
        trans_model_path=f'{model_dir}/ebm_trans.pt',
        rot_model_path=f'{model_dir}/ebm_rot.pt',
        num_negatives_trans=args.num_negatives_trans or cfg.num_neg_trans,
        num_steps_trans=args.num_steps_trans or cfg.num_steps_trans,
        step_size_trans=args.step_size_trans or cfg.step_size_trans,
        step_size_decay_trans=args.step_size_decay_trans or cfg.step_size_decay_trans,
        noise_scale_init_trans=args.noise_scale_init_trans or cfg.noise_scale_init_trans,
        noise_decay_trans=args.noise_decay_trans or cfg.noise_decay_trans,
        
        num_negatives_rot=args.num_negatives_rot or cfg.num_neg_rot,
        num_steps_rot=args.num_steps_rot or cfg.num_steps_rot,
        step_size_rot=args.step_size_rot or cfg.step_size_rot,
        step_size_decay_rot=args.step_size_decay_rot or cfg.step_size_decay_rot,
        noise_scale_init_rot=args.noise_scale_init_rot or cfg.noise_scale_init_rot,
        noise_decay_rot=args.noise_decay_rot or cfg.noise_decay_rot,

        dof_rot=dof_rot,
        dof_trans=dof_trans,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    evaluate_iga_policy(
        task_name=task_name,
        iga_model=iga,
        num_demos=4,
        num_rollouts=10,  # 跑多次rollout测试准确率
        save_video=True
    )
