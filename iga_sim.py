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

from rlbench.tasks import *
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
import queue
import threading

from sim_utils import get_point_cloud, get_rgb_montage_frame, save_video_from_frames
from utils import pose_to_transform, transform_to_pose, subsample_pcd

from iga.models.inference_model import IGA
from iga.utils.parser_utils import get_inference_parser
from iga.utils.common_utils import transform_pcd


# Some examples of RLBench tasks
TASK_NAMES = {
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

def create_sim_env(task_name, headless=False, restrict_rot=True):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES[task_name])

    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    rot_bounds = env._scene.task.base_rotation_bounds()
    mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
    if restrict_rot:
        env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                        (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))
    
    return env, task

def get_demos_with_timeout(task, num_demos, live_demos=True, max_attempts=1, timeout_sec=90):
    """Fetch demos with a timeout to avoid indefinite blocking in RLBench."""
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
    if ok:
        return payload
    raise payload

def deploy_iga_to_rlbench(task_name, iga_model, num_demos=5, save_video=True, video_dir='./videos'):
    env, task = create_sim_env(task_name, headless=True, restrict_rot=True)
    
    gripper_pcd_ee = pickle.load(open('./iga/iga/assets/franka_gripper_pcd.pkl', 'rb'))
    gripper_pcd_ee = subsample_pcd(gripper_pcd_ee, 1024)
    
    print(f"Attempting to fetch {num_demos} offline demo(s)...")
    demos = task.get_demos(num_demos, live_demos=True, max_attempts=10)
    print(f"Successfully fetched {len(demos)} demo(s). Starting deployment...")
    

    print(f"DEBUG: PyRep methods -> {[m for m in dir(task._scene.pyrep) if 'obj' in m.lower()]}")
    all_objs = task._scene.pyrep.get_objects_in_tree()

    # print("--- 场景物体清单 ---")
    # for o in all_objs:
        # name = o.get_name()
        # if 'panda' not in name.lower() and 'camera' not in name.lower():
            # print(f"Name: {name}, Handle: {o.get_handle()}")
    # print("-------------------")

    target_ids = [o.get_handle() for o in all_objs if 'push_button_target' in o.get_name()]
    grasped_ids = [o.get_handle() for o in all_objs if 'item' in o.get_name()]
    print(f"Identified target object IDs: {target_ids}, grasped object IDs: {grasped_ids}")
    
    demo_pcds_grasped = []
    demo_pcds_target = []
    
    for i, d in enumerate(demos):
        if save_video:
            demo_frames = []
            for obs_demo in d:
                frame = get_rgb_montage_frame(obs_demo)
                if frame is not None:
                    demo_frames.append(frame)
            if demo_frames:
                demo_video_path = Path(video_dir) / f'iga_{task_name}_demo_{i}.mp4'
                save_video_from_frames(demo_frames, demo_video_path, fps=15)
                print(f"Saved demo video to: {demo_video_path}")

        final_obs = d[-1]
        final_T_w_e = pose_to_transform(final_obs.gripper_pose)
        
        target_pcd = extract_pcds_from_obs(final_obs, mask_ids=target_ids)

        if grasped_ids:
            print("grasped_ids:", grasped_ids)
            grasped_pcd = extract_pcds_from_obs(final_obs, mask_ids=grasped_ids)
        else:
            grasped_pcd = transform_pcd(gripper_pcd_ee, final_T_w_e)
            
        # 将点云从世界坐标系转换到执行器局部相对坐标系
        inv_final_T_w_e = np.linalg.inv(final_T_w_e)
        grasped_pcd = transform_pcd(grasped_pcd, inv_final_T_w_e)
        target_pcd = transform_pcd(target_pcd, inv_final_T_w_e)
        
        demo_pcds_grasped.append(grasped_pcd)
        demo_pcds_target.append(target_pcd)
        
    demo_pcds = {
        'pcds_grasped': demo_pcds_grasped,
        'pcds_target': demo_pcds_target
    }
    print("Demo data shapes - Grasped:", [pcd.shape for pcd in demo_pcds_grasped], "Target:", [pcd.shape for pcd in demo_pcds_target])

    print("Starting main control loop...")
    descriptions, obs = task.reset()
    max_steps = 30
    
    rollout_frames = []
    if save_video:
        frame = get_rgb_montage_frame(obs)
        if frame is not None:
            rollout_frames.append(frame)
            
    for step in range(max_steps):
        current_T_w_e = pose_to_transform(obs.gripper_pose)
        
        current_target_pcd = extract_pcds_from_obs(obs, mask_ids=target_ids)
        if grasped_ids:
            current_grasped_pcd = extract_pcds_from_obs(obs, mask_ids=grasped_ids)
        else:
            current_grasped_pcd = transform_pcd(gripper_pcd_ee, current_T_w_e)
            
        # 将点云从世界坐标系转换到执行器局部相对坐标系
        inv_current_T_w_e = np.linalg.inv(current_T_w_e)
        current_grasped_pcd = transform_pcd(current_grasped_pcd, inv_current_T_w_e)
        current_target_pcd = transform_pcd(current_target_pcd, inv_current_T_w_e)
            
        live_pcds = {
            'pcd_grasped': current_grasped_pcd,
            'pcd_target': current_target_pcd
        }
        print(f"Step {step} - Live PCD shapes - Grasped: {current_grasped_pcd.shape}, Target: {current_target_pcd.shape}")
        
        T_e_e_new = iga_model.get_transform(
            demo_pcds, 
            live_pcds, 
            visualise=False
        )
        
        from scipy.spatial.transform import Rotation as Rot
        max_trans = 0.03  # 单步最大平移限制：5cm
        max_rot = 0.1     # 单步最大旋转限制：约11.5度
        
        # 判断大动作才插值拆分，小动作直接纳入单次动作
        trans_dist_total = np.linalg.norm(T_e_e_new[:3, 3])
        rot_dist_total = np.linalg.norm(Rot.from_matrix(T_e_e_new[:3, :3]).as_rotvec())

        if trans_dist_total < 0.05 and rot_dist_total > 1.57:
            print("Warning: Ignoring suspicious rotation jump at close range.")
            # only keep the translation, ignore the rotation
            T_e_e_new[:3, :3] = np.eye(3)

        # 设定阈值
        FINE_ALIGN_THRESHOLD = 0.05 # 5cm内进入精细对齐
        gamma = 15.0
        horizon = max(int(trans_dist_total / max_trans), 16)
        k_max = 12
        k_min = 4

        execute_steps = int(np.round(k_min + (k_max - k_min) * (1 - np.exp(-gamma * (trans_dist_total)))))
        print("trans_dist_total:", trans_dist_total, "rot_dist_total:", rot_dist_total, "horizon:", horizon, "execute_steps:", execute_steps)

        actions_list = split_transform_to_horizon(T_e_e_new, horizon=horizon)
        
        original_T_w_e = current_T_w_e
        
        terminate = False
        for T_e_e_interp in actions_list[:execute_steps]:
            # 2. 计算插值的目标位姿
            target_T_w_e_desired = original_T_w_e @ T_e_e_interp 
            
            # 3. 计算相对当前的增量位姿，对其进行步长限幅
            current_T_w_e_actual = pose_to_transform(obs.gripper_pose)
            
            # 当前世界坐标系位姿的逆，左乘到期望的世界坐标系位姿 -> 得到局部增量
            T_delta = np.linalg.inv(current_T_w_e_actual) @ target_T_w_e_desired
            
            trans_diff = T_delta[:3, 3]
            trans_dist = np.linalg.norm(trans_diff)
            if trans_dist > max_trans:
                trans_diff = (trans_diff / trans_dist) * max_trans
                
            rot_diff = Rot.from_matrix(T_delta[:3, :3]).as_rotvec()
            rot_dist = np.linalg.norm(rot_diff)
            if rot_dist > max_rot:
                rot_diff = (rot_diff / rot_dist) * max_rot
                
            T_delta_clipped = np.eye(4)
            T_delta_clipped[:3, :3] = Rot.from_rotvec(rot_diff).as_matrix()
            T_delta_clipped[:3, 3] = trans_diff
            
            # 4. 根据限幅后的增量计算实际上这次需要到达的位姿
            target_T_w_e = current_T_w_e_actual @ T_delta_clipped
            
            target_pose_7d = transform_to_pose(target_T_w_e)
            
            gripper_action = 1.0
            if np.linalg.norm(T_e_e_new[:3, 3]) < 0.5: 
                gripper_action = 0.0
            action = np.append(target_pose_7d, gripper_action)
            
            try:
                obs, reward, terminate = task.step(action)
                if save_video:
                    frame = get_rgb_montage_frame(obs)
                    if frame is not None:
                        rollout_frames.append(frame)
                if terminate:
                    print("Task Completed Successfully!")
                    break
            except Exception as e:
                print(f"Error during step execution: {e}")
                break
                
        if terminate:
            break
            
    if save_video and rollout_frames:
        save_video_from_frames(rollout_frames, Path(video_dir) / f'iga_{task_name}_rollout.mp4', fps=15)
        print(f"Saved deployment video to: iga_{task_name}_rollout.mp4")

    env.shutdown()

def split_transform_to_horizon(T_final, horizon=5):
    """
    将一个 T_final 变换拆解为 horizon 步的小变换序列
    T_final: 4x4 numpy array (相对于当前位姿的增量变换)
    """
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    
    # 1. 提取平移和旋转
    final_trans = T_final[:3, 3]
    final_rot_matrix = T_final[:3, :3]
    final_quat = R.from_matrix(final_rot_matrix).as_quat()
    
    # 初始状态（增量为0，即单位变换）
    start_trans = np.zeros(3)
    start_quat = np.array([0, 0, 0, 1]) # 单位四元数
    
    actions_list = []
    
    key_times = [0, 1]
    key_rots = R.from_quat([start_quat, final_quat])
    slerp = Slerp(key_times, key_rots)
    
    for i in range(1, horizon + 1):
        alpha = i / horizon  # 插值系数 0.2, 0.4, 0.6, 0.8, 1.0
        
        # --- 平移线性插值 ---
        interp_trans = (1 - alpha) * start_trans + alpha * final_trans
        
        # --- 旋转球面线性插值 (SLERP) ---
        interp_rot = slerp(alpha).as_matrix()
        
        # --- 重组为 4x4 矩阵 ---
        T_step = np.eye(4)
        T_step[:3, :3] = interp_rot
        T_step[:3, 3] = interp_trans
        
        actions_list.append(T_step)
        
    return actions_list

# 实现提取点云并在世界坐标系下输出的逻辑
def extract_pcds_from_obs(obs, mask_ids=None):
    pcds = []
    for cam in ('front', 'left_shoulder', 'right_shoulder'):
        cam_pcd = getattr(obs, f'{cam}_point_cloud') # (H, W, 3)
        cam_mask = getattr(obs, f'{cam}_mask')       # (H, W)
        
        mask = np.isin(cam_mask, mask_ids)
        pcds.append(cam_pcd[mask])
            
    pcd = np.concatenate(pcds, axis=0)
    pcd = subsample_pcd(pcd, 1024)
    
    return pcd


if __name__ == '__main__':
    parser = get_inference_parser()
    data_dir = parser.parse_args().data_dir
    model_dir = parser.parse_args().model_dir
    visualise_optimisation = parser.parse_args().visualise_optimisation
    overlay_visualisation = parser.parse_args().overlay_visualisation
    num_neg_trans = parser.parse_args().num_negatives_trans
    num_steps_trans = parser.parse_args().num_steps_trans
    step_size_trans = parser.parse_args().step_size_trans
    step_size_decay_trans = parser.parse_args().step_size_decay_trans
    noise_scale_init_trans = parser.parse_args().noise_scale_init_trans
    noise_decay_trans = parser.parse_args().noise_decay_trans
    num_neg_rot = parser.parse_args().num_negatives_rot
    num_steps_rot = parser.parse_args().num_steps_rot
    step_size_rot = parser.parse_args().step_size_rot
    step_size_decay_rot = parser.parse_args().step_size_decay_rot
    noise_scale_init_rot = parser.parse_args().noise_scale_init_rot
    noise_decay_rot = parser.parse_args().noise_decay_rot

    no_x = parser.parse_args().no_x
    no_y = parser.parse_args().no_y
    no_z = parser.parse_args().no_z
    no_rot_x = parser.parse_args().no_rot_x
    no_rot_y = parser.parse_args().no_rot_y
    no_rot_z = parser.parse_args().no_rot_z

    real_data = parser.parse_args().real_data
    model_dir = './iga/iga/checkpoints'

    iga = IGA(
        trans_model_path=f'{model_dir}/ebm_trans.pt',
        rot_model_path=f'{model_dir}/ebm_rot.pt',
        num_negatives_trans=num_neg_trans,
        num_steps_trans=num_steps_trans,
        step_size_trans=step_size_trans,
        step_size_decay_trans=step_size_decay_trans,
        noise_scale_init_trans=noise_scale_init_trans,
        noise_decay_trans=noise_decay_trans,
        num_negatives_rot=num_neg_rot,
        num_steps_rot=num_steps_rot,
        step_size_rot=step_size_rot,
        step_size_decay_rot=step_size_decay_rot,
        noise_scale_init_rot=noise_scale_init_rot,
        noise_decay_rot=noise_decay_rot,

        dof_rot=(False, False, False, not no_rot_x, not no_rot_y, not no_rot_z),
        dof_trans=(not no_x, not no_y, not no_z, False, False, False),

        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    deploy_iga_to_rlbench(
        task_name='push_button',
        iga_model=iga,
        num_demos=4
    )
 
