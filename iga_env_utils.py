import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from sim_utils import get_rgb_montage_frame, save_video_from_frames
from utils import pose_to_transform, subsample_pcd
from iga.utils.common_utils import transform_pcd

from iga_configs import TASK_NAMES_MAP

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

def extract_pcds_from_obs(obs, mask_ids=None):
    """从观察中提取点云数据"""
    pcds = []
    # Prefer multi-view fusion and include wrist view when available.
    camera_names = ('front', 'left_shoulder', 'right_shoulder', 'wrist')
    for cam in camera_names:
        if not hasattr(obs, f'{cam}_point_cloud') or not hasattr(obs, f'{cam}_mask'):
            continue
        cam_pcd = getattr(obs, f'{cam}_point_cloud')
        cam_mask = getattr(obs, f'{cam}_mask')
        mask = np.isin(cam_mask, mask_ids)
        pcds.append(cam_pcd[mask])
    
    pcd = np.concatenate(pcds, axis=0) if pcds else np.zeros((0,3))
    
    if pcd.shape[0] < 50:
        return None
        
    # pcd = subsample_pcd(pcd, 1024)
    return pcd

def get_object_ids(task, name_list):
    """根据名字关键字获取环境中的物体ID"""
    all_objs = task._scene.pyrep.get_objects_in_tree()
    ids_exact = []
    ids_fallback = []

    # Normalize keywords once to avoid repeated string operations.
    keywords = [k.lower().strip() for k in name_list if k is not None]

    for o in all_objs:
        raw_name = o.get_name().lower()
        # Remove common suffixes and instance postfix for robust exact matching.
        norm_name = raw_name.replace('_visual', '').replace('_respondable', '').split('#')[0].strip()

        if any(norm_name == keyword for keyword in keywords):
            ids_exact.append(o.get_handle())
        elif any(keyword in raw_name for keyword in keywords):
            ids_fallback.append(o.get_handle())

    # Prefer exact matches; fallback to substring matches only when exact is empty.
    return ids_exact if len(ids_exact) > 0 else ids_fallback

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
    """根据两阶段对齐范式提取演示特征, 返回按时间顺序的关键帧列表"""
    # 最终返回的是列表，每个元素是一个 waypoint 字典: {'pcds_grasped': [], 'pcds_target': []}
    demo_waypoints_phase_1 = []
    demo_waypoints_phase_2 = []
    
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = get_object_ids(task, task_cfg.get('phase_1_target', []))
    phase_2_target_ids = get_object_ids(task, task_cfg.get('phase_2_target', []))

    # Keep phase-1 object IDs out of phase-2 targets to avoid source/target contamination.
    phase1_ids_set = set(phase_1_target_ids)
    phase_2_target_ids = [obj_id for obj_id in phase_2_target_ids if obj_id not in phase1_ids_set]

    print(f"Extracted target IDs for phase 1: {phase_1_target_ids}, phase 2: {phase_2_target_ids}")
    
    # Store IDs back in config so rollout can reuse them
    task_cfg['phase_1_target_ids'] = phase_1_target_ids
    task_cfg['phase_2_target_ids'] = phase_2_target_ids

    # 简单处理：这里先假设取第一个成功获取的 demo 来建立关键帧路标
    if len(demos) == 0:
        return demo_waypoints_phase_1, demo_waypoints_phase_2
        
    d = demos[0]

    if save_video and video_dir:
        for i, d_demo in enumerate(demos):
            demo_frames = [get_rgb_montage_frame(o) for o in d_demo if get_rgb_montage_frame(o) is not None]
            if demo_frames:
                demo_video_path = Path(video_dir) / f'iga_{task_name}_demo_{i}.mp4'
                save_video_from_frames(demo_frames, demo_video_path, fps=15)
    
    # 寻找夹爪首次闭合的帧 (或者是抓到物体的帧)
    grasp_idx = len(d) - 1
    if paradigm != 'single_stage':
        for j, o in enumerate(d):
            if o.gripper_open < 0.5:
                grasp_idx = j
                break
                
    # 辅助函数：根据多条演示(demos)提取当前帧对应的特征
    # 考虑到不同demo的长度可能不同，我们简单地以 demos[0] 为时间轴参考
    # 并从其他 demo 找到进度比例相近的帧作为增强特征（可选）
    # 这里为了简单稳健，直接使用单个 demo 提取关键帧，或者把其他 demo 对应进度的点云也放进列表
    
    def extract_wp_dict(phase, current_relative_idx, total_relative_steps):
        # 遍历所有 demo，取进度比例（0~1）相近的帧。这样每个 wp 包含了多个 demo 的信息
        wp_dict = {'pcds_grasped': [], 'pcds_target': []}
        progress = current_relative_idx / max(1, total_relative_steps)
        
        for demo_idx, demo_d in enumerate(demos):
            if phase == 1:
                demo_grasp_idx = len(demo_d) - 1
                if paradigm != 'single_stage':
                    for j, o in enumerate(demo_d):
                        if o.gripper_open < 0.5:
                            demo_grasp_idx = j
                            break
                target_j = int(progress * demo_grasp_idx)
            else:
                demo_grasp_idx = len(demo_d) - 1
                if paradigm != 'single_stage':
                    for j, o in enumerate(demo_d):
                        if o.gripper_open < 0.5:
                            demo_grasp_idx = j
                            break
                target_j = min(len(demo_d)-1, demo_grasp_idx + int(progress * (len(demo_d) - 1 - demo_grasp_idx)))
                
            demo_step_obs = demo_d[target_j]
            step_T_w_e = pose_to_transform(demo_step_obs.gripper_pose)
            
            p_target_pcd = extract_pcds_from_obs(demo_step_obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
            if phase == 1:
                p_source_pcd = transform_pcd(gripper_pcd_ee, step_T_w_e)
            else:
                p_source_pcd = extract_pcds_from_obs(demo_step_obs, mask_ids=phase_1_target_ids)
            
            if p_target_pcd is not None and p_source_pcd is not None:
                inv_step_T_w_e = np.linalg.inv(step_T_w_e)
                p_source_local = transform_pcd(p_source_pcd, inv_step_T_w_e)
                p_target_local = transform_pcd(p_target_pcd, inv_step_T_w_e)
                
                wp_dict['pcds_grasped'].append(p_source_local)
                wp_dict['pcds_target'].append(p_target_local)
                
        return wp_dict

    # 关键帧采样阈值
    dist_threshold = 0.15  # 15cm
    rot_threshold = 0.5    # ~28.6 degrees

    # Phase 1: Reach / Grasp (从开始到抓取帧)
    if paradigm in ['two_stage_pick_and_place', 'two_stage_articulated', 'single_stage']:
        last_saved_T = None
        for step_idx in range(grasp_idx + 1):
            step_obs = d[step_idx]
            step_T_w_e = pose_to_transform(step_obs.gripper_pose)
            
            # 判断是否需要保存为关键帧
            save_kf = False
            if last_saved_T is None or step_idx == grasp_idx:
                save_kf = True
            else:
                T_diff = np.linalg.inv(last_saved_T) @ step_T_w_e
                trans_diff = np.linalg.norm(T_diff[:3, 3])
                rot_diff = np.linalg.norm(Rot.from_matrix(T_diff[:3, :3]).as_rotvec())
                if trans_diff > dist_threshold or rot_diff > rot_threshold:
                    save_kf = True
                    
            if save_kf:
                wp_dict = extract_wp_dict(phase=1, current_relative_idx=step_idx, total_relative_steps=grasp_idx)
                if len(wp_dict['pcds_grasped']) > 0:
                    demo_waypoints_phase_1.append(wp_dict)
                    last_saved_T = step_T_w_e

    # Phase 2: Manipulate / Place (从抓取帧之后到最后)
    if paradigm in ['two_stage_pick_and_place', 'two_stage_articulated']:
        last_saved_T = None
        phase2_total_steps = len(d) - 1 - grasp_idx
        for step_idx in range(grasp_idx, len(d)):
            step_obs = d[step_idx]
            step_T_w_e = pose_to_transform(step_obs.gripper_pose)
            
            save_kf = False
            if last_saved_T is None or step_idx == len(d) - 1:
                save_kf = True
            else:
                T_diff = np.linalg.inv(last_saved_T) @ step_T_w_e
                trans_diff = np.linalg.norm(T_diff[:3, 3])
                rot_diff = np.linalg.norm(Rot.from_matrix(T_diff[:3, :3]).as_rotvec())
                if trans_diff > dist_threshold or rot_diff > rot_threshold:
                    save_kf = True
                    
            if save_kf:
                wp_dict = extract_wp_dict(phase=2, current_relative_idx=step_idx - grasp_idx, total_relative_steps=phase2_total_steps)
                if len(wp_dict['pcds_grasped']) > 0:
                    demo_waypoints_phase_2.append(wp_dict)
                    last_saved_T = step_T_w_e
                
    return demo_waypoints_phase_1, demo_waypoints_phase_2
