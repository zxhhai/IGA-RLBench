from iga.models.inference_model import IGA
import argparse
from iga.utils.common_utils import transform_pcd
from iga.utils.parser_utils import get_inference_parser
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch
from contextlib import contextmanager

LEGEND_INITIAL = 'Legend: Red=Target, Green=Grasped (current)'
LEGEND_FINAL = 'Legend: Red=Target, Yellow=Grasped (before), Green=Grasped (after)'

@contextmanager
def safe_plotting_env():
    # 1. 备份 CoppeliaSim 设置的危险变量
    original_preload = os.environ.get('LD_PRELOAD', '')
    original_platform = os.environ.get('QT_QPA_PLATFORM', '')
    
    # 2. 清除预加载，防止 VTK 调用到 CoppeliaSim 的旧 Qt 库
    if 'LD_PRELOAD' in os.environ:
        del os.environ['LD_PRELOAD']
    
    # 3. 强制设置为无头模式（不依赖 xcb 窗口）
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    try:
        yield
    finally:
        # 4. 绘图结束后恢复环境，确保仿真环境还能跑
        if original_preload:
            os.environ['LD_PRELOAD'] = original_preload
        if original_platform:
            os.environ['QT_QPA_PLATFORM'] = original_platform


def safe_show_plotter(plotter, screenshot_path=None):
    """Show plotter with a fallback for PyVista/VTK backend incompatibilities."""
    renderer = getattr(plotter, 'renderer', None)
    if renderer is not None and not hasattr(renderer, '_actors') and hasattr(renderer, 'actors'):
        try:
            setattr(renderer, '_actors', renderer.actors)
        except Exception:
            renderer_cls = renderer.__class__
            if not hasattr(renderer_cls, '_actors'):
                try:
                    setattr(renderer_cls, '_actors', property(lambda self: self.actors))
                except Exception:
                    pass

    show_kwargs = {}
    if screenshot_path is not None:
        show_kwargs = {'screenshot': screenshot_path, 'auto_close': False, 'interactive': False}
    else:
        show_kwargs = {'jupyter_backend': 'static'}

    try:
        plotter.show(**show_kwargs)
    except AttributeError as err:
        if "_actors" not in str(err):
            raise
        # Fallback for versions where Renderer private attributes differ.
        fallback_kwargs = dict(show_kwargs)
        fallback_kwargs.pop('jupyter_backend', None)
        fallback_kwargs.setdefault('auto_close', False)
        fallback_kwargs.setdefault('interactive', False)
        plotter.show(**fallback_kwargs)


def vis_initial_poses(pcd_target, pcd_grasped, save_path=None):
    iga.visualiser.clear()
    iga.visualiser.add_pcd_to_plotter(pcd_target * iga.scaling_factor_global,
                                      name='live_pcd_target', color='red',
                                      radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.add_pcd_to_plotter(pcd_grasped * iga.scaling_factor_global,
                                      name='live_pcd_grasped', color='green',
                                      radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.safe_add_text('Initial Object Poses, press q to continue.',
                                 font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.safe_add_text(LEGEND_INITIAL,
                                 font_size=12, position='lower_left', color='k')
    safe_show_plotter(iga.visualiser.plotter, screenshot_path=save_path)
    iga.visualiser.safe_add_text('Optimising.',
                                 font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.clear()


def vis_final_poses(pcd_target, pcd_grasped, T_e_e_new, save_path=None):
    iga.visualiser.clear()
    iga.visualiser.add_pcd_to_plotter(pcd_target * iga.scaling_factor_global,
                                      name='live_pcd_target', color='red',
                                      radius=0.003 * iga.scaling_factor_global)

    iga.visualiser.add_pcd_to_plotter(pcd_grasped * iga.scaling_factor_global,
                                      name='live_pcd_grasped_old', color='yellow',
                                      radius=0.003 * iga.scaling_factor_global, opacity=0.5)
    iga.visualiser.add_pcd_to_plotter(
        transform_pcd(pcd_grasped, T_e_e_new) * iga.scaling_factor_global,
        name='live_pcd_grasped', color='green',
        radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.safe_add_text('Object Poses After Optimisation. Yellow - Initial Pose. press q to continue.',
                                 font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.safe_add_text(LEGEND_FINAL,
                                 font_size=12, position='lower_left', color='k')
    safe_show_plotter(iga.visualiser.plotter, screenshot_path=save_path)
    iga.visualiser.clear()


if __name__ == '__main__':
    ####################################################################################################################
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
    enable_visualisation = visualise_optimisation != ''
    enable_graph_visualisation = visualise_optimisation == 'graph'
    save_vis_dir = './vis_outputs'
    os.makedirs(save_vis_dir, exist_ok=True)
    ####################################################################################################################
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
    ####################################################################################################################
    num_samples = len([file for file in os.listdir(data_dir) if 'sample' in file and file.endswith('.pkl')])
    gripper_pcd = pickle.load(open('./assets/franka_gripper_pcd.pkl', 'rb'))
    for k in range(num_samples):
        raw_sample = pickle.load(open(f'{data_dir}/sample_{k}.pkl', 'rb'))
        sample = {
            'demo_pcds': {
                'pcds_grasped': [raw_sample['pcds_a'][i] if raw_sample['pcds_a'][i] is not None else gripper_pcd for i
                                 in range(1, len(raw_sample['pcds_a']))],
                'pcds_target': raw_sample['pcds_b'][1:]},
            'live_pcds': {
                'pcd_grasped': raw_sample['pcds_a'][0] if raw_sample['pcds_a'][0] is not None else gripper_pcd,
                'pcd_target': raw_sample['pcds_b'][0]}
        }
        if not real_data:
            # Applying a random transformation to the live observations to simulate the real world.
            T_rand = np.eye(4)
            T_rand[:3, :3] = Rot.random().as_matrix()
            T_rand[:3, 3] = np.random.uniform(-0.3, 0.3, 3)
            sample['live_pcds']['pcd_target'] = transform_pcd(sample['live_pcds']['pcd_target'], T_rand)
        ################################################################################################################
        if enable_visualisation:
            initial_vis_path = os.path.join(save_vis_dir, f'sample_{k:04d}_initial.png')
            with safe_plotting_env():
                vis_initial_poses(sample['live_pcds']['pcd_target'], sample['live_pcds']['pcd_grasped'],
                                  save_path=initial_vis_path)
        ################################################################################################################
        T_e_e_new = iga.get_transform(sample['demo_pcds'],
                                      sample['live_pcds'],
                                      visualise=enable_visualisation,
                                      overlay=overlay_visualisation,
                                      vis_graph=enable_graph_visualisation)
        ################################################################################################################
        if enable_visualisation:
            final_vis_path = os.path.join(save_vis_dir, f'sample_{k:04d}_final.png')
            with safe_plotting_env():
                vis_final_poses(sample['live_pcds']['pcd_target'], sample['live_pcds']['pcd_grasped'], T_e_e_new,
                                save_path=final_vis_path)
    ####################################################################################################################
