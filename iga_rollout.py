import os
import pickle
import json
import csv
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as Rot

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from pyrep.objects.proximity_sensor import ProximitySensor

from sim_utils import get_rgb_montage_frame, save_video_from_frames
from utils import pose_to_transform, transform_to_pose, subsample_pcd
from iga.utils.common_utils import transform_pcd

from iga_configs import IGAConfig, get_task_config
from iga_env_utils import create_sim_env, extract_pcds_from_obs, split_transform_to_horizon, extract_demo_features, get_object_ids

LEGEND_INITIAL = 'Legend: Red=Live Target, Green=Live Grasped (current)'
LEGEND_FINAL = 'Legend: Red=Live Target, Yellow=Live Grasped (before), Green=Live Grasped (after)'
LEGEND_DEMO = 'Legend: Blue=Demo Target, Cyan=Demo Grasped'


def _pcd_count(pcd):
    return 0 if pcd is None else int(len(pcd))


def _safe_float(x):
    value = float(x)
    return value if np.isfinite(value) else np.nan


def _collect_demo_stats(demo_pcds):
    if demo_pcds is None:
        return {
            'demo_target_sets': 0,
            'demo_grasped_sets': 0,
            'demo_target_points_mean': 0.0,
            'demo_grasped_points_mean': 0.0,
        }
    target_counts = [len(x) for x in demo_pcds.get('pcds_target', []) if x is not None]
    grasped_counts = [len(x) for x in demo_pcds.get('pcds_grasped', []) if x is not None]
    return {
        'demo_target_sets': int(len(target_counts)),
        'demo_grasped_sets': int(len(grasped_counts)),
        'demo_target_points_mean': float(np.mean(target_counts)) if target_counts else 0.0,
        'demo_grasped_points_mean': float(np.mean(grasped_counts)) if grasped_counts else 0.0,
    }


def _save_debug_curves(debug_events, debug_dir, rollout_idx):
    if plt is None:
        return

    action_events = [e for e in debug_events if e.get('event') == 'action_step']
    waypoint_events = [e for e in debug_events if e.get('event') == 'waypoint_prediction']
    if not action_events and not waypoint_events:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=140)

    if action_events:
        xs = [e['global_step'] for e in action_events]
        axes[0].plot(xs, [_safe_float(e.get('dist_to_obj', np.nan)) for e in action_events], label='dist_to_obj')
        axes[0].plot(xs, [_safe_float(e.get('dist_to_target', np.nan)) for e in action_events], label='dist_to_target')
        axes[0].plot(xs, [_safe_float(e.get('delta_trans', np.nan)) for e in action_events], label='delta_trans')
        axes[0].plot(xs, [_safe_float(e.get('delta_rot', np.nan)) for e in action_events], label='delta_rot')
        axes[0].set_title('Action Step Metrics')
        axes[0].set_ylabel('Distance / Delta')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

    if waypoint_events:
        xs = [e['global_step'] for e in waypoint_events]
        axes[1].plot(xs, [_safe_float(e.get('pred_trans', np.nan)) for e in waypoint_events], label='pred_trans')
        axes[1].plot(xs, [_safe_float(e.get('pred_rot', np.nan)) for e in waypoint_events], label='pred_rot')
        axes[1].plot(xs, [_safe_float(e.get('horizon', np.nan)) for e in waypoint_events], label='horizon')
        axes[1].set_title('Waypoint Prediction Metrics')
        axes[1].set_xlabel('Global Step')
        axes[1].set_ylabel('Prediction')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, f'rollout_{rollout_idx:03d}_debug_curves.png'))
    plt.close(fig)


def export_rollout_debug(debug_events, debug_dir, rollout_idx):
    if not debug_events:
        return
    os.makedirs(debug_dir, exist_ok=True)

    jsonl_path = os.path.join(debug_dir, f'rollout_{rollout_idx:03d}_debug_events.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for event in debug_events:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    csv_path = os.path.join(debug_dir, f'rollout_{rollout_idx:03d}_debug_events.csv')
    fields = sorted({key for event in debug_events for key in event.keys()})
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in debug_events:
            writer.writerow(event)

    _save_debug_curves(debug_events, debug_dir, rollout_idx)


@contextmanager
def safe_plotting_env():
    original_preload = os.environ.get('LD_PRELOAD', '')
    original_platform = os.environ.get('QT_QPA_PLATFORM', '')

    if 'LD_PRELOAD' in os.environ:
        del os.environ['LD_PRELOAD']
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    try:
        yield
    finally:
        if original_preload:
            os.environ['LD_PRELOAD'] = original_preload
        if original_platform:
            os.environ['QT_QPA_PLATFORM'] = original_platform


def _subsample_points(points, max_points=2500):
    if points is None:
        return None
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def _save_fast_3d_snapshot(path, title, legend_text, point_sets):
    """Save a fast non-interactive 3D snapshot via matplotlib Agg backend."""
    if plt is None:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    x_vals = []
    y_vals = []
    z_vals = []
    for pts, color, size, alpha, label in point_sets:
        if pts is None or len(pts) == 0:
            continue
        pts_np = _subsample_points(pts)
        if pts_np is None or pts_np.ndim != 2 or pts_np.shape[1] < 3:
            continue
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], c=color, s=size, alpha=alpha, label=label)
        x_vals.append(pts_np[:, 0])
        y_vals.append(pts_np[:, 1])
        z_vals.append(pts_np[:, 2])

    if x_vals and y_vals and z_vals:
        x_all = np.concatenate(x_vals)
        y_all = np.concatenate(y_vals)
        z_all = np.concatenate(z_vals)
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        z_min, z_max = float(np.min(z_all)), float(np.max(z_all))
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)
        half = 0.55 * max_range
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)

    ax.view_init(elev=24, azim=42)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.2)
    fig.text(0.01, 0.01, legend_text, fontsize=9, va='bottom', ha='left')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_rollout_alignment_snapshot(iga_model, pcd_target, pcd_grasped, save_path, text, T_e_e_new=None, demo_pcds=None):
    scale = iga_model.scaling_factor_global
    pcd_target_scaled = pcd_target * scale
    pcd_grasped_scaled = pcd_grasped * scale

    if T_e_e_new is None:
        live_sets = [
            (pcd_target_scaled, 'red', 4, 0.8, 'Live Target'),
            (pcd_grasped_scaled, 'green', 4, 0.8, 'Live Grasped (current)'),
        ]
        legend_text = LEGEND_INITIAL
    else:
        pcd_grasped_after = transform_pcd(pcd_grasped, T_e_e_new) * scale
        live_sets = [
            (pcd_target_scaled, 'red', 4, 0.8, 'Live Target'),
            (pcd_grasped_scaled, 'yellow', 4, 0.5, 'Live Grasped (before)'),
            (pcd_grasped_after, 'green', 4, 0.8, 'Live Grasped (after)'),
        ]
        legend_text = LEGEND_FINAL

    _save_fast_3d_snapshot(
        path=save_path,
        title=text,
        legend_text=legend_text,
        point_sets=live_sets,
    )

    if demo_pcds is not None:
        demo_targets = demo_pcds.get('pcds_target', [])
        demo_graspeds = demo_pcds.get('pcds_grasped', [])
        demo_path = f"{os.path.splitext(save_path)[0]}_demo{os.path.splitext(save_path)[1]}"
        demo_sets = []
        for i, (demo_target, demo_grasped) in enumerate(zip(demo_targets, demo_graspeds)):
            demo_sets.append((demo_target * scale, 'blue', 3, 0.35, f'Demo Target {i}'))
            demo_sets.append((demo_grasped * scale, 'cyan', 3, 0.35, f'Demo Grasped {i}'))

        _save_fast_3d_snapshot(
            path=demo_path,
            title=f'{text} (Demo Only)',
            legend_text=LEGEND_DEMO,
            point_sets=demo_sets,
        )


class AutoGraspController:
    def __init__(self, gripper, proximity_sensor_name='Panda_gripper_attachProxSensor', 
                 dist_threshold=0.015, wait_steps=10):
        """
        :param gripper: RLBench 的 Gripper 对象
        :param proximity_sensor_name: 仿真器中挂在夹爪中心感应器的名称
        :param dist_threshold: 触发抓取的平移距离阈值 (meters)
        :param wait_steps: 闭合夹爪后强制等待的物理步数，确保抓稳
        """
        self.gripper = gripper
        # 获取夹爪中心的感应器，这是判断“物体包络”最准的方法
        try:
            self.prox_sensor = ProximitySensor(proximity_sensor_name)
        except Exception:
            self.prox_sensor = None
        
        self.dist_threshold = dist_threshold
        self.wait_steps = wait_steps
        self.is_grasped = False
        self.stepping_counter = 0

    def _check_envelope(self, target_obj):
        """
        判断物体是否在夹爪的包络空间内
        方式 A: 使用仿真器的 Proximity Sensor (推荐)
        方式 B: 计算物体中心是否在夹爪两个手指的 Bounding Box 之间
        """
        if self.prox_sensor is None:
            return True # Fallback if sensor not found
        # 检测感应器范围内是否有特定物体
        try:
            return self.prox_sensor.is_detected(target_obj)
        except Exception:
            return True

    def decide_action(self, ee_pose, target_pose, target_obj):
        """
        根据当前状态决定夹爪动作
        ee_pose: [x, y, z, qx, qy, qz, qw]
        """
        dist = np.linalg.norm(ee_pose[:3] - target_pose[:3])
        
        # 逻辑 1：如果还未抓取，且距离足够近，且物体在包络内
        if not self.is_grasped:
            in_envelope = self._check_envelope(target_obj)
            
            if dist < self.dist_threshold and in_envelope:
                print(f"[AutoGrasp] Conditions met: Dist={dist:.4f}, InEnvelope={in_envelope}. Closing.")
                self.is_grasped = True
                self.stepping_counter = self.wait_steps
                return 0.0  # 0.0 代表闭合指令
            return 1.0      # 保持开启
        
        # 逻辑 2：如果已经抓取，处于等待稳定期
        if self.is_grasped and self.stepping_counter > 0:
            self.stepping_counter -= 1
            return 0.0
            
        # 逻辑 3：抓取完成后的维持状态
        return 0.0

    def reset(self):
        self.is_grasped = False
        self.stepping_counter = 0

def get_gripper_action(config, dist_to_obj, dist_to_target, is_grasped):
    mode = config.get('gripper_mode', 'constant')
    
    if mode == 'constant':
        return config.get('gripper_value', 1.0)
        
    if mode == 'proximity_grasp':
        if is_grasped:
            return 0.0
        return 1.0
        
    if mode == 'pick_and_place':
        if not is_grasped:
            return 1.0
        else:
            # 已经抓到了，判断是否到达终点该放手了
            return 1.0 if dist_to_target < config.get('dist_threshold', 0.05) else 0.0
            
    return 1.0

def run_rollout(iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, task_cfg, save_video=False, rollout_idx=0, cfg=None, cheat_demo=None):
    """两阶段单次rollout测试 (支持 Oracle Grasping)"""
    if cfg is None: cfg = IGAConfig()
        
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = task_cfg.get('phase_1_target_ids', [])
    phase_2_target_ids = task_cfg.get('phase_2_target_ids', [])
    iga_vis_mode = task_cfg.get('iga_visualise_optimisation', 'pcd')
    enable_iga_visualisation = iga_vis_mode != ''
    iga_vis_graph = iga_vis_mode == 'graph'
    iga_overlay_visualisation = task_cfg.get('iga_overlay_visualisation', False)
    iga_save_visualisation = task_cfg.get('iga_save_visualisation', True)
    iga_save_vis_dir = task_cfg.get('iga_save_vis_dir', './iga_rollout_vis')
    iga_save_initial_final = task_cfg.get('iga_save_initial_final', True)
    iga_initial_final_dir = os.path.join(iga_save_vis_dir, 'initial_final')

    debug_enabled = task_cfg.get('debug_enabled', True)
    debug_save_step_vis = task_cfg.get('debug_save_step_vis', False)
    debug_step_vis_interval = max(1, int(task_cfg.get('debug_step_vis_interval', 20)))
    debug_dir_root = task_cfg.get('debug_dir', './iga_debug')
    rollout_debug_dir = os.path.join(debug_dir_root, f'rollout_{rollout_idx:03d}')
    debug_step_vis_dir = os.path.join(rollout_debug_dir, 'step_vis')
    debug_events = []
    global_step_counter = 0

    if debug_enabled:
        os.makedirs(rollout_debug_dir, exist_ok=True)
        if debug_save_step_vis:
            os.makedirs(debug_step_vis_dir, exist_ok=True)

    def log_debug(event_type, **kwargs):
        if not debug_enabled:
            return
        event = {
            'event': event_type,
            'rollout_idx': int(rollout_idx),
        }
        event.update(kwargs)
        debug_events.append(event)

    def flush_debug():
        if debug_enabled:
            export_rollout_debug(debug_events, rollout_debug_dir, rollout_idx)

    if enable_iga_visualisation and iga_save_visualisation:
        os.makedirs(iga_save_vis_dir, exist_ok=True)
        if iga_save_initial_final:
            os.makedirs(iga_initial_final_dir, exist_ok=True)
        iga_model.visualiser.save_dir = iga_save_vis_dir
        if getattr(iga_model.visualiser, 'config', None) is not None:
            iga_model.visualiser.config['record'] = True
    else:
        if getattr(iga_model.visualiser, 'config', None) is not None:
            iga_model.visualiser.config['record'] = False
    
    max_steps_per_phase = cfg.max_steps_per_phase
    rollout_frames = []
    success = False
    
    if cheat_demo is not None:
        cheat_demo.restore_state() # RLBench 中恢复 demo 状态
        try:
            descriptions, obs = task.reset_to_demo(cheat_demo)
        except AttributeError:
            descriptions, obs = task.reset()
            try: obs = task.get_observation()
            except: pass
    else:
        descriptions, obs = task.reset()

    # Initialize AutoGraspController
    try:
        gripper_obj = task._robot.gripper 
    except AttributeError:
        gripper_obj = None
    # dist_threshold = task_cfg.get('dist_threshold', 0.05)
    grasp_ctrl = AutoGraspController(gripper_obj, dist_threshold=0.02, wait_steps=20)
    
    if save_video:
        f = get_rgb_montage_frame(obs)
        if f is not None: rollout_frames.append(f)

    if cheat_demo is not None:
        print("Replaying Phase 1 (Oracle Grasping) from Demo...")
        for step_idx in range(1, len(cheat_demo)):
            step_obs = cheat_demo[step_idx]
            
            # 提取动作
            action_pose = step_obs.gripper_pose
            action_gripper = [1.0] if step_obs.gripper_open > 0.5 else [0.0]
            action = np.concatenate([action_pose, action_gripper])
            
            try:
                obs, reward, terminate = task.step(action)
            except Exception:
                pass
                
            if step_obs.gripper_open < 0.5:
                print("Grasp achieved in Demo Replay! Handing over to IGA model for Phase 2.")
                # 抓到后多停留几步确保抓稳物理状态，延长停留时间到 20 步，并保持末端位置不动
                try:
                    current_pose = obs.gripper_pose
                except AttributeError:
                    current_pose = action_pose
                
                stabilize_action = np.concatenate([current_pose, [0.0]])
                for _ in range(20):
                    try:
                        obs, _, _ = task.step(stabilize_action)
                        if save_video:
                            f = get_rgb_montage_frame(obs)
                            if f is not None: rollout_frames.append(f)
                    except: pass
                    
                # 标记抓取器为已抓取状态，防止 phase 2 初始化时判断错误
                grasp_ctrl.is_grasped = True
                grasp_ctrl.stepping_counter = 0
                break
            
    def run_phase(phase, max_steps, demo_waypoints=None):
        nonlocal obs, success, rollout_frames, global_step_counter
        
        # 假设 demo_waypoints 是一个当前阶段按顺序排列的关键帧列表
        if demo_waypoints is None:
            demo_waypoints = [demo_pcds_p1 if phase == 1 else demo_pcds_p2]
            
        total_wps = len(demo_waypoints)
        log_debug('phase_start', phase=int(phase), total_waypoints=int(total_wps), max_steps=int(max_steps))

        if phase == 2 and len(phase_2_target_ids) == 0:
            print('[WARN] phase_2_target_ids is empty; phase 2 target extraction will fail.')
            log_debug('warning', phase=int(phase), message='phase_2_target_ids_empty')

        last_valid_target_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
        if last_valid_target_pcd is None: last_valid_target_pcd = np.zeros((1024, 3))
        
        last_valid_source_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids) if phase == 2 else None
        if phase == 2 and last_valid_source_pcd is None: last_valid_source_pcd = np.zeros((1024, 3))

        last_stable_T_w_target = None
        total_executed_steps = 0
        current_wp_idx = 0

        while current_wp_idx < total_wps:
            if total_executed_steps >= max_steps:
                print(f"[Phase {phase}] Max steps reached.")
                log_debug('phase_max_steps_reached', phase=int(phase), executed_steps=int(total_executed_steps))
                break
                
            current_demo_pcd = demo_waypoints[current_wp_idx]
            current_T_w_e = pose_to_transform(obs.gripper_pose)
            
            # 1. Update Target
            current_target_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
                
            # 2. Update Source
            if phase == 1:
                source_pcd = transform_pcd(gripper_pcd_ee, current_T_w_e)
            else:
                source_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids)
                
            is_pcd_stable = (current_target_pcd is not None) and (phase == 1 or source_pcd is not None)
            demo_stats = _collect_demo_stats(current_demo_pcd)
            log_debug(
                'waypoint_observation',
                phase=int(phase),
                waypoint_idx=int(current_wp_idx),
                global_step=int(global_step_counter),
                pcd_stable=int(bool(is_pcd_stable)),
                live_target_points=int(_pcd_count(current_target_pcd)),
                live_source_points=int(_pcd_count(source_pcd)),
                **demo_stats,
            )
            
            if is_pcd_stable:
                last_valid_target_pcd = current_target_pcd
                if phase == 2:
                    last_valid_source_pcd = source_pcd
                    
                inv_current_T_w_e = np.linalg.inv(current_T_w_e)
                source_pcd_local = transform_pcd(source_pcd, inv_current_T_w_e)
                current_target_pcd_local = transform_pcd(current_target_pcd, inv_current_T_w_e)
                source_center = np.mean(source_pcd_local, axis=0)
                target_center = np.mean(current_target_pcd_local, axis=0)
                center_dist = float(np.linalg.norm(source_center - target_center))
                overlap_threshold = float(task_cfg.get('debug_overlap_center_threshold', 0.02))
                overlap_warning = int(center_dist < overlap_threshold)
                if overlap_warning:
                    print(f"[WARN] Possible source/target overlap at phase {phase}, wp {current_wp_idx}: center_dist={center_dist:.4f}")

                log_debug(
                    'waypoint_alignment_input',
                    phase=int(phase),
                    waypoint_idx=int(current_wp_idx),
                    global_step=int(global_step_counter),
                    center_dist=float(center_dist),
                    overlap_warning=int(overlap_warning),
                )
                    
                live_pcds = {'pcd_grasped': source_pcd_local, 'pcd_target': current_target_pcd_local}

                if enable_iga_visualisation and iga_save_visualisation and iga_save_initial_final:
                    initial_path = os.path.join(
                        iga_initial_final_dir,
                        f'rollout_{rollout_idx:03d}_phase_{phase}_wp_{current_wp_idx:03d}_step_{total_executed_steps:04d}_initial.png',
                    )
                    with safe_plotting_env():
                        save_rollout_alignment_snapshot(
                            iga_model,
                            pcd_target=current_target_pcd_local,
                            pcd_grasped=source_pcd_local,
                            save_path=initial_path,
                            text='Rollout Initial Pose',
                            demo_pcds=current_demo_pcd,
                        )
                
                # 仅预测一次当前路标点的对齐变换
                T_e_e_new = iga_model.get_transform(
                    current_demo_pcd,
                    live_pcds,
                    visualise=False,
                    overlay=False,
                    vis_graph=False,
                )

                if enable_iga_visualisation and iga_save_visualisation and iga_save_initial_final:
                    final_path = os.path.join(
                        iga_initial_final_dir,
                        f'rollout_{rollout_idx:03d}_phase_{phase}_wp_{current_wp_idx:03d}_step_{total_executed_steps:04d}_final.png',
                    )
                    with safe_plotting_env():
                        save_rollout_alignment_snapshot(
                            iga_model,
                            pcd_target=current_target_pcd_local,
                            pcd_grasped=source_pcd_local,
                            save_path=final_path,
                            text='Rollout Final Pose',
                            T_e_e_new=T_e_e_new,
                            demo_pcds=current_demo_pcd,
                        )
                last_stable_T_w_target = current_T_w_e @ T_e_e_new
            else:
                if last_stable_T_w_target is not None:
                    T_e_e_new = np.linalg.inv(current_T_w_e) @ last_stable_T_w_target
                    log_debug(
                        'waypoint_fallback_last_stable',
                        phase=int(phase),
                        waypoint_idx=int(current_wp_idx),
                        global_step=int(global_step_counter),
                    )
                else:
                    T_e_e_new = np.eye(4)
                    log_debug(
                        'waypoint_fallback_identity',
                        phase=int(phase),
                        waypoint_idx=int(current_wp_idx),
                        global_step=int(global_step_counter),
                    )
            
            trans_dist_total = np.linalg.norm(T_e_e_new[:3, 3])
            rot_dist_total = np.linalg.norm(Rot.from_matrix(T_e_e_new[:3, :3]).as_rotvec())

            # if trans_dist_total < 0.05 and rot_dist_total > 1.57:
                # print("Warning: Ignoring suspicious rotation jump at close range.")
                # T_e_e_new[:3, :3] = np.eye(3)
                # rot_dist_total = 0.0

            # --- 收敛判定（Convergence Check）---
            # 如果机器人当前状态和该 Waypoint 的距离足够小，说明该节点已经到达或误差极小，可以直接前进
            # if trans_dist_total < 0.02 :
                # print(f"[Phase {phase}] Converged to Waypoint {current_wp_idx+1}/{total_wps}")
                # current_wp_idx += 1
                # continue

            horizon = max(int(trans_dist_total / cfg.max_trans), int(rot_dist_total / cfg.max_rot), cfg.horizon_min)
            log_debug(
                'waypoint_prediction',
                phase=int(phase),
                waypoint_idx=int(current_wp_idx),
                global_step=int(global_step_counter),
                pred_trans=float(trans_dist_total),
                pred_rot=float(rot_dist_total),
                horizon=int(horizon),
            )
            
            print(f"[Phase {phase}] Tracking Waypoint {current_wp_idx+1}/{total_wps} | trans_dist: {trans_dist_total:.4f}, rot_dist: {rot_dist_total:.4f}, horizon: {horizon}")
            
            # 直接使用完整的动作序列
            actions_list = split_transform_to_horizon(T_e_e_new, horizon=horizon)
            original_T_w_e = current_T_w_e
            
            for T_e_e_interp in actions_list:
                if total_executed_steps >= max_steps:
                    break
                total_executed_steps += 1
                global_step_counter += 1
                
                current_T_w_e_actual = pose_to_transform(obs.gripper_pose)
                is_grasped = obs.gripper_open < 0.5
                
                # 动态计算抓取逻辑的距离
                p1_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids)
                p1_center = np.mean(p1_pcd, axis=0) if p1_pcd is not None and p1_pcd.shape[0] > 0 else np.array([float('inf'), float('inf'), float('inf')])
                dist_to_obj = np.linalg.norm(current_T_w_e_actual[:3, 3] - p1_center)

                p2_pcd = extract_pcds_from_obs(obs, mask_ids=phase_2_target_ids)
                p2_center_active = np.mean(p2_pcd, axis=0) if p2_pcd is not None and p2_pcd.shape[0] > 0 else np.array([float('inf'), float('inf'), float('inf')])
                dist_to_target = np.linalg.norm(current_T_w_e_actual[:3, 3] - p2_center_active)
                
                current_ee_pose_6d = transform_to_pose(current_T_w_e_actual)
                target_pose_6d = transform_to_pose(original_T_w_e @ T_e_e_new)
                
                if task_cfg.get('gripper_mode', 'constant') == 'proximity_grasp' or task_cfg.get('gripper_mode', 'pick_and_place') == 'pick_and_place':
                    if phase == 1:
                        gripper_action = grasp_ctrl.decide_action(current_ee_pose_6d, target_pose_6d, target_obj=None)
                    else:
                        gripper_action = get_gripper_action(task_cfg, dist_to_obj, dist_to_target, is_grasped)
                else:
                    gripper_action = get_gripper_action(task_cfg, dist_to_obj, dist_to_target, is_grasped)
                
                # 如果正在闭合夹爪（等待稳定抓稳期），则强制末端停止运动，单纯闭合夹爪
                if phase == 1 and grasp_ctrl.is_grasped and grasp_ctrl.stepping_counter > 0:
                    target_T_w_e_desired = current_T_w_e_actual
                else:
                    target_T_w_e_desired = original_T_w_e @ T_e_e_interp 
                
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

                # 修正四元数符号：确保与当前末端位姿方向一致，避免 IK 误判为大幅旋转
                current_quat = obs.gripper_pose[3:]
                if np.dot(current_quat, target_pose_7d[3:]) < 0:
                    target_pose_7d[3:] = -target_pose_7d[3:]

                action = np.append(target_pose_7d, gripper_action)
                log_debug(
                    'action_step',
                    phase=int(phase),
                    waypoint_idx=int(current_wp_idx),
                    local_step=int(total_executed_steps),
                    global_step=int(global_step_counter),
                    dist_to_obj=float(dist_to_obj),
                    dist_to_target=float(dist_to_target),
                    delta_trans=float(trans_dist),
                    delta_rot=float(rot_dist),
                    gripper_action=float(gripper_action),
                    is_grasped=int(bool(is_grasped)),
                )

                if debug_enabled and debug_save_step_vis and (global_step_counter % debug_step_vis_interval == 0):
                    target_world = p1_pcd if phase == 1 else p2_pcd
                    source_world = transform_pcd(gripper_pcd_ee, current_T_w_e_actual) if phase == 1 else p1_pcd
                    if target_world is not None and source_world is not None:
                        inv_current_T_w_e_actual = np.linalg.inv(current_T_w_e_actual)
                        step_target_local = transform_pcd(target_world, inv_current_T_w_e_actual)
                        step_source_local = transform_pcd(source_world, inv_current_T_w_e_actual)
                        step_vis_path = os.path.join(
                            debug_step_vis_dir,
                            f'phase_{phase}_wp_{current_wp_idx:03d}_gstep_{global_step_counter:05d}.png',
                        )
                        with safe_plotting_env():
                            save_rollout_alignment_snapshot(
                                iga_model,
                                pcd_target=step_target_local,
                                pcd_grasped=step_source_local,
                                save_path=step_vis_path,
                                text=f'Step Debug P{phase} W{current_wp_idx} G{global_step_counter}',
                                T_e_e_new=T_delta_clipped,
                                demo_pcds=current_demo_pcd,
                            )
                
                try:
                    obs, reward, terminate = task.step(action)
                    if save_video:
                        f = get_rgb_montage_frame(obs)
                        if f is not None: rollout_frames.append(f)
                    if terminate:
                        log_debug('terminate', phase=int(phase), global_step=int(global_step_counter), success=1)
                        success = True
                        return True
                except Exception as e:
                    print(f"Step execution failed with error: {e}. Continuing with next action.")
                    log_debug('step_exception', phase=int(phase), global_step=int(global_step_counter), message=str(e))
                    pass
                    
                if phase == 1 and grasp_ctrl.is_grasped and grasp_ctrl.stepping_counter == 0:
                    print(f"[Phase Switch] Grasp completed. Exiting Phase 1.")
                    log_debug('phase_switch_after_grasp', phase=int(phase), global_step=int(global_step_counter))
                    return False
                
            current_wp_idx += 1
                
        return False
        
    # Execution Flow
    if paradigm == 'single_stage':
        if cheat_demo is None:
            # 兼容原来单个点云模式，你可以将 demo_pcds_p1 变为列表
            run_phase(1, max_steps_per_phase, demo_waypoints=demo_pcds_p1 if isinstance(demo_pcds_p1, list) else [demo_pcds_p1])
    elif paradigm in ['two_stage_pick_and_place', 'two_stage_articulated']:
        if cheat_demo is None:
            # Phase 1: Reach
            run_phase(1, max_steps_per_phase, demo_waypoints=demo_pcds_p1 if isinstance(demo_pcds_p1, list) else [demo_pcds_p1])
            if success:
                flush_debug()
                return True, rollout_frames
            
            # Grasp Action Transition (根据当前模式自适应)
            gripper_action_idx = get_gripper_action(task_cfg, dist_to_obj=0.0, dist_to_target=float('inf'), is_grasped=True)
            action_pose = transform_to_pose(pose_to_transform(obs.gripper_pose))
            action = np.append(action_pose, gripper_action_idx)
            terminate = False
            try:
                obs, reward, terminate = task.step(action)
            except Exception:
                pass
                
            if save_video:
                f = get_rgb_montage_frame(obs)
                if f is not None: rollout_frames.append(f)
                
            if terminate:
                success = True
                log_debug('terminate_after_transition_step', phase=1, global_step=int(global_step_counter), success=1)
                flush_debug()
                return True, rollout_frames
            
        # Phase 2: Manipulate/Place
        run_phase(2, max_steps_per_phase, demo_waypoints=demo_pcds_p2 if isinstance(demo_pcds_p2, list) else [demo_pcds_p2])
    log_debug('rollout_end', success=int(bool(success)), total_global_steps=int(global_step_counter))
    flush_debug()
    return success, rollout_frames



def evaluate_iga_policy(task_name, iga_model, num_demos=4, num_rollouts=10, save_video=True, video_dir='./videos', cache_dir='./demo_cache'):
    """使用给定的Demo，测试多次rollout以计算成功率"""
    video_dir = str(Path(video_dir) / task_name)
    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    task_cfg = get_task_config(task_name)
    task_cfg['task_name'] = task_name
    cfg = IGAConfig()
    env, task = create_sim_env(task_name, headless=True, restrict_rot=task_cfg['restrict_rot'])

    gripper_pcd_ee = pickle.load(open('./iga/iga/assets/franka_gripper_pcd.pkl', 'rb'))
    # gripper_pcd_ee = subsample_pcd(gripper_pcd_ee, 1024)

    # --- Demo 缓存：避免每次重新采样和提取特征 ---
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / f'{task_name}_n{num_demos}.pkl'

    # handle ID 每次 env.launch() 后可能重新分配，永远实时获取，不从缓存读
    task_cfg['phase_1_target_ids'] = get_object_ids(task, task_cfg.get('phase_1_target', []))
    task_cfg['phase_2_target_ids'] = get_object_ids(task, task_cfg.get('phase_2_target', []))

    # Avoid source/target overlap: phase-2 targets must exclude phase-1 object IDs.
    phase1_ids_set = set(task_cfg['phase_1_target_ids'])
    task_cfg['phase_2_target_ids'] = [
        obj_id for obj_id in task_cfg['phase_2_target_ids'] if obj_id not in phase1_ids_set
    ]
    print(f"Object IDs: phase1={task_cfg['phase_1_target_ids']}, phase2={task_cfg['phase_2_target_ids']}")
    if task_cfg.get('paradigm', 'two_stage_pick_and_place') != 'single_stage' and len(task_cfg['phase_2_target_ids']) == 0:
        print('[WARN] phase_2_target_ids is empty after overlap filtering. Check phase_2_target keywords in task config.')

    if cache_path.exists():
        print(f"Loading cached demo features from {cache_path} ...")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        demo_pcds_p1 = cache['demo_pcds_p1']
        demo_pcds_p2 = cache['demo_pcds_p2']
        print(f"Loaded {len(demo_pcds_p1)} phase-1 waypoints, {len(demo_pcds_p2)} phase-2 waypoints.")
    else:
        print(f"Fetching {num_demos} offline demo(s)...")
        demos = task.get_demos(num_demos, live_demos=True, max_attempts=50)
        demo_pcds_p1, demo_pcds_p2 = extract_demo_features(demos, gripper_pcd_ee, task, task_cfg, save_video, video_dir, task_name)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'demo_pcds_p1': demo_pcds_p1,
                'demo_pcds_p2': demo_pcds_p2,
            }, f)
        print(f"Demo features cached to {cache_path}.")

    success_count = 0
    print(f"Starting {num_rollouts} evaluation rollouts for task: {task_name}")


    for i in range(num_rollouts):
        print(f"--- Rollout {i+1}/{num_rollouts} ---")
        try:
            cd = task.get_demos(1, live_demos=True, random_selection=True)
            cheat_demo = cd[0] if len(cd) > 0 else None
        except Exception as e:
            print(f"Failed to get cheat demo: {e}")
            cheat_demo = None
    
        success, frames = run_rollout(
            iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, 
            task_cfg, save_video, rollout_idx=i, cfg=cfg, 
            cheat_demo=cheat_demo
        )
        
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
