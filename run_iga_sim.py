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

import torch
from iga.models.inference_model import IGA
from iga.utils.parser_utils import get_inference_parser

# Import extracted modules
from iga_configs import IGAConfig, get_task_config
from iga_rollout import evaluate_iga_policy


import numpy as np
from iga_configs import TASK_NAMES_MAP
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig



def get_name_list():
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, './', obs_config=obs_config, headless=True)
    env.launch()
    
    for task_name, task_cls in TASK_NAMES_MAP.items():
        print(f"==========================================")
        print(f"Task: {task_name}")
        print(f"==========================================")
        
        try:
            task = env.get_task(task_cls)
            task.reset()
            
            objs = task._scene.pyrep.get_objects_in_tree()
            
            # 过滤掉机械臂、相机、光源、环境墙壁边界和内置Dummy点等非任务关键物体
            exclude_keys = ['panda', 'camera', 'vision', 'sensor', 'light', 'floor', 
                            'wall', 'ceiling', 'dummy', 'waypoint', 'boundary', 
                            'default', 'joint', 'link', 'workspace']
            
            names = []
            for o in objs:
                name = o.get_name()
                if not any(k in name.lower() for k in exclude_keys):
                    # 去掉底层网格的后缀(如 _visual, _respondable)及序号(#0)，只保留核心名字
                    clean_name = name.replace('_visual', '').replace('_respondable', '').split('#')[0]
                    names.append(clean_name)
            
            # 去重并排序
            names = sorted(list(set(names)))
            print(", ".join(names))
            
        except Exception as e:
            print(f"Error loading task {task_name}: {e}")
            
        print("\n")
        
    env.shutdown()

def get_target_ids():
    from iga_env_utils import get_object_ids
    
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, './', obs_config=obs_config, headless=True)
    env.launch()
    
    for task_name, task_cls in TASK_NAMES_MAP.items():
        print(f"==========================================")
        print(f"Task: {task_name}")
        print(f"==========================================")
        
        try:
            task = env.get_task(task_cls)
            task.reset()
            task_cfg = get_task_config(task_name)
            
            phase_1_targets = task_cfg.get('phase_1_target', [])
            phase_2_targets = task_cfg.get('phase_2_target', [])
            
            p1_ids = get_object_ids(task, phase_1_targets)
            p2_ids = get_object_ids(task, phase_2_targets)
            
            print(f"Phase 1 target ({phase_1_targets}): {p1_ids}")
            print(f"Phase 2 target ({phase_2_targets}): {p2_ids}")
            
        except Exception as e:
            print(f"Error loading task {task_name}: {e}")
            
        print("\n")

    env.shutdown()

if __name__ == '__main__':
    # get_target_ids()
    get_name_list()
    parser = get_inference_parser()
    # Add task argument directly here so it works even if not defined in the imported parser
    parser.add_argument('--task', type=str, default='push_button', help='Name of the RLBench task to run')
    parser.add_argument('--num_demos', type=int, default=4, help='Number of demonstration trajectories to use')
    parser.add_argument('--num_rollouts', type=int, default=10, help='Number of rollout trajectories to use')
    args, unknown = parser.parse_known_args()
    
    # 获取任务名称以自适应调整参数
    task_name = args.task
    num_demos = args.num_demos
    num_rollouts = args.num_rollouts
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
        num_demos=num_demos,
        num_rollouts=num_rollouts,
        save_video=True
    )
