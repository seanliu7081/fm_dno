#!/usr/bin/env python3
"""Calibrate task-image intended-displacement overlays from original LIBERO XML.

The dataset stores vertically flipped OpenGL images. This script reconstructs
only camera/kinematics, never integrates actions. HDF5 `states[t]` predates its
stored observation, so the observed robot joint and gripper positions replace
robot qpos before validating the gripper position. Background/object state comes
from `states[t+1]`; residual replay differences are measured, not concealed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import zarr


def float32_action_digest(actions: np.ndarray) -> str:
    arr = np.ascontiguousarray(actions, dtype='<f4')
    return hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()


def project_world(points: np.ndarray, world_to_pixel: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    projected = homogeneous @ world_to_pixel.T
    return projected[..., :2] / projected[..., 2:3], projected[..., 2]


def camera_transform(position, rotation, fovy, height, width):
    """World -> homogeneous image with x right, y down and positive depth."""
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation).reshape(3, 3)
    pose[:3, 3] = position
    pose = pose @ np.diag([1., -1., -1., 1.])
    focal = .5 * height / np.tan(np.deg2rad(fovy) / 2.)
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    return intrinsics @ np.linalg.inv(pose)


def relocated_xml(xml: str, libero_assets: Path):
    robosuite_root = Path(importlib.util.find_spec('robosuite').origin).parent
    tree = ET.fromstring(xml)
    changes = {}
    for element in tree.iter():
        original = element.get('file')
        if original is None:
            continue
        if '/robosuite/' in original:
            target = robosuite_root / original.rsplit('/robosuite/', 1)[1]
        elif '/assets/' in original:
            target = libero_assets / original.split('/assets/', 1)[1]
        else:
            target = Path(original)
        target = target.resolve()
        if not target.is_file():
            raise FileNotFoundError(f'Missing demonstration asset: {original} -> {target}')
        element.set('file', str(target))
        changes[original] = str(target)
    return ET.tostring(tree, encoding='unicode'), changes


def demonstration_index(raw_dir: Path):
    index = {}
    for path in sorted(raw_dir.glob('*.hdf5')):
        with h5py.File(path, 'r') as handle:
            for demo_name, demo in handle['data'].items():
                digest = float32_action_digest(demo['actions'][:])
                index.setdefault(digest, []).append((path, demo_name))
    return index


def match_episodes(zarr_root, selection, raw_dir):
    ends = zarr_root['meta/episode_ends'][:]
    starts = np.r_[0, ends[:-1]]
    index = demonstration_index(raw_dir)
    matches = {}
    for episode in sorted({int(w['episode_index']) for w in selection['windows']}):
        start, end = int(starts[episode]), int(ends[episode])
        actions = zarr_root['data/action'][start:end]
        digest = float32_action_digest(actions)
        candidates = index.get(digest, [])
        if len(candidates) != 1:
            raise RuntimeError(f'Episode {episode}: expected unique exact action hash, got {len(candidates)}')
        path, demo_name = candidates[0]
        with h5py.File(path, 'r') as handle:
            demo = handle[f'data/{demo_name}']
            assert np.array_equal(actions, demo['actions'][:].astype(np.float32))
            assert np.array_equal(zarr_root['data/robot0_eef_pos'][start:end], demo['obs/ee_pos'][:].astype(np.float32))
            assert np.array_equal(zarr_root['data/robot0_joint_pos'][start:end], demo['obs/joint_states'][:].astype(np.float32))
            frames = sorted({int(w['frame_index']) for w in selection['windows'] if int(w['episode_index']) == episode})
            for frame in frames:
                expected = np.flip(demo['obs/agentview_rgb'][frame], axis=0)
                assert np.array_equal(zarr_root['data/agentview_rgb'][start + frame], expected), (episode, frame)
            assert path.stem[:-len('_demo')] == next(w['task_name'] for w in selection['windows'] if int(w['episode_index']) == episode)
        matches[episode] = {'episode_index': episode, 'hdf5_path': str(path), 'demo_name': demo_name,
                            'action_sha256_float32': digest, 'episode_start': start, 'episode_end': end,
                            'all_actions_exact_match': True, 'all_eef_positions_exact_match': True,
                            'all_joint_positions_exact_match': True, 'verified_image_frames': frames,
                            'images_exact_match_after_vertical_flip': True}
    return matches


def restore_state(mj, model, state):
    data = mj.MjData(model)
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[1 + model.nq + model.nv:1 + model.nq + model.nv + model.na]
    mj.mj_forward(model, data)
    return data


def restore_observed_robot(mj, model, data, joints, gripper):
    for i, value in enumerate(joints):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'robot0_joint{i + 1}')
        if joint_id < 0:
            raise RuntimeError(f'Missing Panda joint {i + 1}')
        data.qpos[model.jnt_qposadr[joint_id]] = value
    for i, value in enumerate(gripper):
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'gripper0_finger_joint{i + 1}')
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mj.mj_forward(model, data)


def plot_alignment(output, images, renders, pixels, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for start in range(0, len(images), 6):
        stop = min(start + 6, len(images))
        fig, axes = plt.subplots(stop - start, 3, figsize=(9, 2.8 * (stop - start)), squeeze=False)
        for row, idx in enumerate(range(start, stop)):
            rec = records[idx]
            axes[row, 0].imshow(images[idx]); axes[row, 1].imshow(renders[idx])
            axes[row, 2].imshow(np.abs(images[idx].astype(float) - renders[idx].astype(float)).mean(-1), cmap='magma', vmin=0, vmax=80)
            for ax in axes[row, :2]:
                ax.plot(*pixels[idx], marker='+', ms=14, mew=1.5, color='#00ffdd')
            axes[row, 0].set_title(f"R{idx:02d}: task {rec['task_uid']} / {rec['phase']} / dataset")
            axes[row, 1].set_title(f"Restored robot; EEF error {rec['eef_kinematic_error_m']:.2g} m")
            axes[row, 2].set_title(f"Mean pixel difference {rec['render_image_mae']:.2f}/255")
            for ax in axes[row]:
                ax.axis('off')
        fig.suptitle('Camera alignment audit: cyan cross is projected recorded gripper position', fontsize=12)
        fig.tight_layout()
        fig.savefig(output / f'projection_alignment_{start // 6 + 1:02d}.png', dpi=150)
        plt.close(fig)


def plot_contact_sheet(output, images, renders, pixels):
    from PIL import Image, ImageDraw
    count = len(images)
    rows = (count + 2) // 3
    canvas = Image.new('RGB', (6 * 128, rows * 148), 'white')
    draw = ImageDraw.Draw(canvas)
    for idx in range(count):
        row, phase = divmod(idx, 3)
        for column, source in enumerate((images, renders)):
            x, y = (phase * 2 + column) * 128, row * 148
            canvas.paste(Image.fromarray(source[idx]), (x, y + 20))
            label = f"T{row} " + ('early', 'middle', 'late')[phase] + (' data' if column == 0 else ' sim')
            draw.text((x + 2, y + 3), label, fill='black')
            px, py = pixels[idx]; px += x; py += y + 20
            draw.line((px - 5, py, px + 5, py), fill='#00ffdd', width=1)
            draw.line((px, py - 5, px, py + 5), fill='#00ffdd', width=1)
    canvas.save(output / 'projection_contact_sheet.jpg', quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/workspace/fm_dno/data/libero/libero10_N500.zarr'))
    parser.add_argument('--raw-dir', type=Path, default=Path('/workspace/past_action/third_party/LIBERO/libero/datasets/libero_10'))
    parser.add_argument('--libero-assets', type=Path, default=Path('/workspace/fm_dno/third_party/LIBERO/libero/libero/assets'))
    parser.add_argument('--egl-device', type=int, default=7)
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)
    import mujoco as mj

    output = args.output_dir.resolve()
    selection = json.loads((output / 'selection.json').read_text())
    comparison_data = np.load(output / 'data.npz', allow_pickle=False) if (output / 'data.npz').exists() else None
    zarr_root = zarr.open(str(args.dataset), mode='r')
    matches = match_episodes(zarr_root, selection, args.raw_dir)
    (output / 'hdf5_episode_matches.json').write_text(json.dumps(list(matches.values()), indent=2) + '\n')
    records, transforms, images, renders, ee_positions, ee_pixels = [], [], [], [], [], []
    controller_scales = []
    representatives = sorted(selection['representatives'], key=lambda r: r['representative_index'])
    assert [int(r['representative_index']) for r in representatives] == list(range(len(representatives)))
    for rep in representatives:
        match = matches[int(rep['episode_index'])]
        frame, global_index = int(rep['frame_index']), int(rep['global_index'])
        assert global_index == match['episode_start'] + frame
        with h5py.File(match['hdf5_path'], 'r') as handle:
            demo = handle[f"data/{match['demo_name']}"]
            original_xml = demo.attrs['model_file']
            xml, asset_map = relocated_xml(original_xml, args.libero_assets)
            model = mj.MjModel.from_xml_string(xml)
            # Same-index saved state is PRE action; the image and proprioception are POST action.
            state_index = min(frame + 1, len(demo['states']) - 1)
            data = restore_state(mj, model, demo['states'][state_index])
            eef_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'gripper0_grip_site')
            eef = demo['obs/ee_pos'][frame]
            next_state_eef_error = float(np.linalg.norm(data.site_xpos[eef_id] - eef))
            restore_observed_robot(mj, model, data, demo['obs/joint_states'][frame], demo['obs/gripper_states'][frame])
            error = float(np.linalg.norm(data.site_xpos[eef_id] - eef))
            if error > .005:
                raise RuntimeError(f'Observation EEF kinematics do not align: {rep} error {error}m')
            image = zarr_root['data/agentview_rgb'][global_index]
            if comparison_data is not None:
                assert np.array_equal(comparison_data['obs__agentview_rgb'][int(rep['window_index']), -1], image)
                assert np.array_equal(comparison_data['obs__robot0_eef_pos'][int(rep['window_index']), -1], eef.astype(np.float32))
            height, width = image.shape[:2]
            camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'agentview')
            transform = camera_transform(data.cam_xpos[camera_id], data.cam_xmat[camera_id], model.cam_fovy[camera_id], height, width)
            pixel, depth = project_world(eef, transform)
            assert np.linalg.norm(project_world(data.site_xpos[eef_id], transform)[0] - pixel) < .5, 'Recorded eef and reconstructed gripper disagree by >= 0.5 pixel'
            assert depth > 0 and 0 <= pixel[0] < width and 0 <= pixel[1] < height, (rep, pixel, depth)
            with mj.Renderer(model, height=height, width=width) as renderer:
                options = mj.MjvOption()
                options.geomgroup[0] = 0  # Robosuite default: render visuals, suppress collision geometry.
                options.geomgroup[1] = 1
                renderer.update_scene(data, camera='agentview', scene_option=options)
                render = renderer.render().copy()  # Native mujoco Renderer already returns upright pixels.
            mae = float(np.abs(image.astype(float) - render.astype(float)).mean())
            flipped_mae = float(np.abs(image.astype(float) - render[::-1].astype(float)).mean())
            if not mae < flipped_mae:
                raise RuntimeError(f'Image orientation check failed for {rep}: {mae} vs flip {flipped_mae}')
            kwargs = json.loads(handle['data'].attrs['env_args'])['env_kwargs']
            controller = kwargs['controller_configs']
            assert controller['type'] == 'OSC_POSE' and controller['control_delta'] is True
            assert controller['input_min'] == -1 and controller['input_max'] == 1
            assert np.allclose(controller['output_max'][:3], .05)
            assert np.allclose(controller['output_min'][:3], -.05)
            controller_scales.append(float(controller['output_max'][0]))
            records.append({**rep, 'hdf5_path': match['hdf5_path'], 'demo_name': match['demo_name'],
                            'camera_name': 'agentview', 'image_height': height, 'image_width': width,
                            'camera_fovy_degrees': float(model.cam_fovy[camera_id]),
                            'camera_position_world': data.cam_xpos[camera_id].tolist(),
                            'projected_eef_xy_pixels': pixel.tolist(), 'eef_camera_depth_m': float(depth),
                            'eef_kinematic_error_m': error, 'eef_kinematic_error_pixels': float(np.linalg.norm(project_world(data.site_xpos[eef_id], transform)[0] - pixel)), 'saved_next_state_eef_error_m': next_state_eef_error,
                            'render_image_mae': mae, 'vertically_flipped_render_image_mae': flipped_mae,
                            'orientation_check_passed': True, 'controller_config': controller,
                            'source_xml_sha256': hashlib.sha256(original_xml.encode()).hexdigest(),
                            'relocated_asset_count': len(asset_map), 'object_state_index': state_index,
                            'robot_state_source': 'HDF5 same-frame observed joint_states and gripper_states'})
            transforms.append(transform); images.append(image); renders.append(render); ee_positions.append(eef); ee_pixels.append(pixel)
        print(f"calibrated R{rep['representative_index']:02d} task {rep['task_uid']} {rep['phase']}: EEF={error:.3g}m image MAE={mae:.2f} vs flip={flipped_mae:.2f}", flush=True)
    assert np.allclose(controller_scales, .05)
    np.savez_compressed(output / 'projection.npz', world_to_pixel=np.stack(transforms), ee_pos=np.stack(ee_positions),
                        image=np.stack(images), controller_translation_scale=np.array(.05),
                        controller_input_min=np.array(-1.), controller_input_max=np.array(1.),
                        projected_eef_xy=np.stack(ee_pixels), reconstructed_image=np.stack(renders),
                        representative_window_indices=np.array([r['window_index'] for r in representatives]))
    audit = {'schema_version': 1, 'mujoco_version': mj.__version__, 'representatives': records,
             'dataset': str(args.dataset.resolve()), 'raw_dataset_dir': str(args.raw_dir.resolve()),
             'matched_episode_count': len(matches), 'source_hdf5_index_matches_all_selected_windows': True,
             'comparison_data_current_observations_exactly_verified': comparison_data is not None,
             'world_to_pixel_convention': 'p_h = [world_x,world_y,world_z,1] @ M.T; pixel_xy = p_h[:2] / p_h[2]; depth=p_h[2]>0',
             'image_orientation': 'x right, y down; zarr is vertical-flipped source HDF5 OpenGL image; native mujoco Renderer already upright',
             'camera_model': 'XML camera fovy, runtime camera world pose, centered pinhole intrinsics matching robosuite camera_utils',
             'controller_translation_scale_m_per_unit': .05, 'controller_input_clip': [-1., 1.],
             'heatmap_semantics': 'Endpoint density after cumulative first 8 clipped XYZ action commands times 0.05 m, starting at observed EEF position. INTENDED displacement only; no dynamics or robot rollout.',
             'rendering_caveat': 'Plain XML reconstruction may omit runtime logical material changes, such as a glowing activated stove. Stored zarr images are used for all final heatmaps; reconstructed renders are alignment audits only.',
             'eef_alignment_tolerance_pixels': .5,
             'dataset_timing_caveat': 'Source LIBERO create_dataset.py stores post-action images/proprioception alongside same-index pre-action states and actions. Calibration uses exact observed robot joints; object rendering uses saved successor state. Evaluation retains the training dataset indexing.',
             'all_eef_kinematic_checks_passed': all(r['eef_kinematic_error_pixels'] < .5 for r in records),
             'all_image_orientation_checks_passed': all(r['orientation_check_passed'] for r in records),
             'max_eef_kinematic_error_m': max(r['eef_kinematic_error_m'] for r in records), 'max_eef_kinematic_error_pixels': max(r['eef_kinematic_error_pixels'] for r in records),
             'max_reconstruction_image_mae': max(r['render_image_mae'] for r in records)}
    (output / 'projection.json').write_text(json.dumps(audit, indent=2) + '\n')
    plot_alignment(output, images, renders, ee_pixels, records)
    plot_contact_sheet(output, images, renders, ee_pixels)
    print(f'Saved projection.npz and verified {len(records)} representative camera calibrations', flush=True)


if __name__ == '__main__':
    main()
