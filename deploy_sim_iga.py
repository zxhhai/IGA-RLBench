import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iga'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instant_policy'))

import pickle
import numpy as np
import argparse
import torch
from pathlib import Path
from tqdm import tqdm, trange

from iga.models.inference_model import IGA
from iga.utils.parser_utils import get_inference_parser
from sim_utils import (
    create_sim_env, 
    get_demos_with_timeout,
    get_point_cloud,
    get_rgb_montage_frame,
    save_video_from_frames
)
from iga.utils import transform_pcd, subsample_pcd, pose_to_transform, transform_to_pose


def parse_args():
    parser = get_inference_parser()
    
    # RLBench Simulation args
    parser.add_argument('--task_name', type=str, default='phone_on_base', help='RLBench task name')
    parser.add_argument('--num_demos', type=int, default=1, help='Number of demos to use for conditioning')
    parser.add_argument('--num_rollouts', type=int, default=10, help='Number of evaluation rollouts')
    parser.add_argument('--max_execution_steps', type=int, default=30, help='Max execution steps per rollout')
    parser.add_argument('--headless', action='store_true', help='Run CoppeliaSim in headless mode')
    parser.add_argument('--save_video', action='store_true', help='Save video of rollouts')
    parser.add_argument('--video_dir', type=str, default='./videos', help='Directory to save videos')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize IGA Model
    iga = IGA(
        trans_model_path=f'{args.model_dir}/ebm_trans.pt',
        rot_model_path=f'{args.model_dir}/ebm_rot.pt',
        num_negatives_trans=args.num_negatives_trans,
        num_steps_trans=args.num_steps_trans,
        step_size_trans=args.step_size_trans,
        step_size_decay_trans=args.step_size_decay_trans,
        noise_scale_init_trans=args.noise_scale_init_trans,
        noise_decay_trans=args.noise_decay_trans,
        num_negatives_rot=args.num_negatives_rot,
        num_steps_rot=args.num_steps_rot,
        step_size_rot=args.step_size_rot,
        step_size_decay_rot=args.step_size_decay_rot,
        noise_scale_init_rot=args.noise_scale_init_rot,
        noise_decay_rot=args.noise_decay_rot,
        dof_rot=(False, False, False, not args.no_rot_x, not args.no_rot_y, not args.no_rot_z),
        dof_trans=(not args.no_x, not args.no_y, not args.no_z, False, False, False),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load default gripper pcd
    gripper_pcd_path = './iga/assets/franka_gripper_pcd.pkl'
    if os.path.exists(gripper_pcd_path):
        gripper_pcd = pickle.load(open(gripper_pcd_path, 'rb'))
    else:
        gripper_pcd = np.zeros((1024, 3))
        
    # Create Sim Env
    env, task = create_sim_env(args.task_name, headless=args.headless, restrict_rot=True)
    print(f"Task name: {task.get_name()}")
    
    # Collect offline demo for demonstrations
    print("Attempting to fetch offline demos...")
    demos = get_demos_with_timeout(task, args.num_demos, live_demos=True, max_attempts=5, timeout_sec=90)
    
    # Process demo into IGA format constraint
    demo_pcds_grasped = []
    demo_pcds_target = []
    
    for obs in demos[0]:  # Use the first demo
        scene_pcd = get_point_cloud(obs)
        # Assuming scene_pcd acts as target context
        demo_pcds_target.append(subsample_pcd(scene_pcd, 1024) if len(scene_pcd)>0 else gripper_pcd)
        demo_pcds_grasped.append(gripper_pcd)  # Simplified: constantly use gripper pcd as grasped object

    demo_pcds = {
        'pcds_grasped': demo_pcds_grasped[1:],
        'pcds_target': demo_pcds_target[1:]
    }
    
    # Rollout execution loop
    successes = []
    pbar = trange(args.num_rollouts, desc=f'Evaluating model, SR: 0/{args.num_rollouts}', leave=True)
    
    for i in pbar:
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except:
                continue

        success = 0
        rollout_frames = []
        
        for k in range(args.max_execution_steps):
            curr_obs = task.get_observation()
            if args.save_video:
                frame = get_rgb_montage_frame(curr_obs)
                if frame is not None:
                    rollout_frames.append(frame)
                    
            # Current pose and point cloud
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            scene_pcd = get_point_cloud(curr_obs)
            
            live_pcd_target = subsample_pcd(scene_pcd, 1024) if len(scene_pcd)>0 else gripper_pcd
            live_pcds = {
                'pcd_grasped': gripper_pcd,
                'pcd_target': live_pcd_target
            }
            
            # Predict delta transformation using IGA
            with torch.no_grad():
                T_e_e_new = iga.get_transform(demo_pcds,
                                              live_pcds,
                                              visualise=False,
                                              overlay=False,
                                              vis_graph=False)
            
            env_action = np.zeros(8)
            # Compose transformations: new pose in world frame
            T_w_e_new = T_w_e @ T_e_e_new
            env_action[:7] = transform_to_pose(T_w_e_new)
            env_action[7] = 1.0 # default gripper open (1)
            
            try:
                curr_obs, reward, terminate = task.step(env_action)
                success = int(terminate and reward > 0.)
            except Exception as e:
                terminate = True
                
            if terminate:
                break

        successes.append(success)
        if args.save_video:
            save_video_from_frames(rollout_frames, Path(args.video_dir) / f'iga_rollout_{i+1}.mp4', fps=15)
            
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        
    pbar.close()
    env.shutdown()
    print(f"Final Success Rate: {sum(successes) / len(successes):.2f}")

if __name__ == '__main__':
    main()
