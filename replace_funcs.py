import re

with open('/root/autodl-tmp/run_iga_sim.py', 'r') as f:
    text = f.read()

# Define new versions of get_gripper_action, extract_demo_features, run_rollout
# And evaluate_iga_policy

new_funcs = """def extract_demo_features(demos, gripper_pcd_ee, task, task_cfg, save_video=False, video_dir=None, task_name="task"):
    '''根据两阶段对齐范式提取演示特征'''
    demo_pcds_phase_1 = {'pcds_grasped': [], 'pcds_target': []}
    demo_pcds_phase_2 = {'pcds_grasped': [], 'pcds_target': []}
    
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = get_object_ids(task, task_cfg.get('phase_1_target', []))
    phase_2_target_ids = get_object_ids(task, task_cfg.get('phase_2_target', []))
    
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
            # The target of Phase 1 is the item to grasp or interact with.
            # And the source is the gripper itself.
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
            # The target of Phase 2 is the final destination.
            # And the source is the object we already grasped in Phase 1 (i.e. phase_1_target).
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

def get_gripper_action(config, current_ee_pose, phase_2_target_pos, is_object_grasped=False, current_phase=1):
    mode = config.get('gripper_mode', 'constant')
    
    if mode == 'constant':
        return config.get('gripper_value', 1.0)
        
    if current_phase == 1:
        # Phase 1: Keep open to reach the object, close if proximity logic applies but ideally let the phase transition handle it.
        if is_object_grasped:
            return 0.0
        return 1.0
        
    elif current_phase == 2:
        dist = np.linalg.norm(current_ee_pose[:3, 3] - phase_2_target_pos) if phase_2_target_pos is not None else float('inf')
        if mode == 'proximity_grasp':
            # Once grasped, keep grasped
            return 0.0
        elif mode == 'pick_and_place':
            # Release when close to phase 2 target
            return 1.0 if dist < config.get('dist_threshold', 0.05) else 0.0
            
    return 1.0

def run_rollout(iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, task_cfg, save_video=False, rollout_idx=0, cfg=None):
    '''两阶段单次rollout测试'''
    if cfg is None: cfg = IGAConfig()
        
    paradigm = task_cfg.get('paradigm', 'two_stage_pick_and_place')
    phase_1_target_ids = task_cfg.get('phase_1_target_ids', [])
    phase_2_target_ids = task_cfg.get('phase_2_target_ids', [])
    
    descriptions, obs = task.reset()
    
    max_steps_per_phase = 20
    rollout_frames = []
    
    if save_video:
        f = get_rgb_montage_frame(obs)
        if f is not None: rollout_frames.append(f)
            
    success = False
    
    def run_phase(phase, max_steps):
        nonlocal obs, success, rollout_frames
        
        last_valid_target_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids if phase == 1 else phase_2_target_ids)
        if last_valid_target_pcd is None: last_valid_target_pcd = np.zeros((1024, 3))
        
        if phase == 1:
            source_pcds_demo = demo_pcds_p1
        else:
            source_pcds_demo = demo_pcds_p2
            last_valid_source_pcd = extract_pcds_from_obs(obs, mask_ids=phase_1_target_ids)
            if last_valid_source_pcd is None: last_valid_source_pcd = np.zeros((1024, 3))
            
            # Identify Phase 2 Target Pos for Gripper Logic
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

            horizon = max(int(trans_dist_total / cfg.max_trans), cfg.horizon_min)
            execute_steps = int(np.round(cfg.k_min + (cfg.k_max - cfg.k_min) * (1 - np.exp(-cfg.gamma * trans_dist_total))))
            
            relative_actions = interpolate_trajectory(T_e_e_new, horizon)[:execute_steps]
            
            is_grasped = obs.gripper_open < 0.5
            p2_center_val = p2_center if phase == 2 and 'p2_center' in locals() else None
            gripper_action = get_gripper_action(task_cfg, current_T_w_e, p2_center_val, is_object_grasped=is_grasped, current_phase=phase)
            
            for T_step in relative_actions:
                current_T_w_e_actual = pose_to_transform(obs.gripper_pose)
                target_T_w_e = current_T_w_e_actual @ T_step
                
                action_pose = transform_to_pose(target_T_w_e)
                action = np.append(action_pose, gripper_action)
                
                obs, reward, terminate = task.step(action)
                if save_video:
                    f = get_rgb_montage_frame(obs)
                    if f is not None: rollout_frames.append(f)
                    
                if task.success()[0]:
                    success = True
                    return True
                    
            if trans_dist_total < cfg.fine_align_threshold and rot_dist_total < 0.1:
                # Close enough, phase complete
                break
                
        return False
        
    # Execution Flow
    if paradigm == 'single_stage':
        run_phase(1, max_steps_per_phase)
    elif paradigm in ['two_stage_pick_and_place', 'two_stage_articulated']:
        # Phase 1: Reach
        run_phase(1, max_steps_per_phase)
        if success: return True, rollout_frames
        
        # Grasp Action
        action_pose = obs.gripper_pose
        action = np.append(action_pose, 0.0) # Close gripper
        # Give simulation time to settle the grasp
        for _ in range(5):
            task._scene.step()
        obs, reward, terminate = task.step(action)
        if save_video:
            f = get_rgb_montage_frame(obs)
            if f is not None: rollout_frames.append(f)
            
        if task.success()[0]:
            success = True
            return True, rollout_frames
            
        # Phase 2: Manipulate/Place
        run_phase(2, max_steps_per_phase)
        
    return success, rollout_frames

def evaluate_iga_policy(iga_model, task_name, num_demos=1, num_rollouts=10, save_video=False):
    '''策略主测试循环'''
    video_dir = f'./videos/{task_name}' if save_video else None
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        
    task_cfg = get_task_config(task_name)
    cfg = IGAConfig()
    env, task = create_sim_env(task_name, headless=True, restrict_rot=task_cfg.get('restrict_rot', False))
    
    gripper_pcd_ee = pickle.load(open('./iga/iga/assets/franka_gripper_pcd.pkl', 'rb'))
    gripper_pcd_ee = subsample_pcd(gripper_pcd_ee, 1024)
    
    print(f"Fetcing {num_demos} offline demo(s)...")
    demos = task.get_demos(num_demos, live_demos=True, max_attempts=50)
    
    demo_pcds_p1, demo_pcds_p2 = extract_demo_features(demos, gripper_pcd_ee, task, task_cfg, save_video, video_dir, task_name)
    
    success_count = 0
    for idx in range(num_rollouts):
        print(f"Starting rollout {idx+1}/{num_rollouts} for {task_name}...")
        success, frames = run_rollout(iga_model, task, demo_pcds_p1, demo_pcds_p2, gripper_pcd_ee, task_cfg, save_video, idx, cfg)
        
        if success:
            success_count += 1
            print(f"  -> Rollout {idx+1} SUCCESS")
        else:
            print(f"  -> Rollout {idx+1} FAILED")
            
        if save_video and frames:
            status = "success" if success else "fail"
            video_path = Path(video_dir) / f'iga_{task_name}_rollout_{idx}_{status}.mp4'
            save_video_from_frames(frames, video_path, fps=15)
            
    print(f"[{task_name}] Final Success Rate: {success_count}/{num_rollouts} ({(success_count/num_rollouts)*100:.1f}%)")
    env.shutdown()
    return success_count
"""

# Regex out from extract_demo_features up to the end (excluding the if __name__)
pattern = re.compile(r'def extract_demo_features\(.*?\):.*?env\.shutdown\(\)\n    return success_count', re.DOTALL)
new_text = pattern.sub(new_funcs.strip(), text)

with open('/root/autodl-tmp/run_iga_sim.py', 'w') as f:
    f.write(new_text)
