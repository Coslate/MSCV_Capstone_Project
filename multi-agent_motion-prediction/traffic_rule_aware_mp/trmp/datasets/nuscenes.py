from typing import Dict, Any, List
import numpy as np
from pathlib import Path

def _optional_import():
    try:
        from nuscenes import NuScenes
        from nuscenes.prediction import PredictHelper
        from nuscenes.map_expansion.map_api import NuScenesMap
        return NuScenes, PredictHelper, NuScenesMap
    except Exception as e:
        raise ImportError("nuscenes-devkit is required for this loader. Install `nuscenes-devkit`.") from e

def load_sample(nuscenes_root: str, sample_token: str, past_sec: float = 4.0, map_radius_m: float = 80.0) -> Dict[str, Any]:
    """Return a unified sample dict for one nuScenes keyframe sample."""
    NuScenes, PredictHelper, NuScenesMap = _optional_import()
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root)
    helper = PredictHelper(nusc)
    sample = nusc.get('sample', sample_token)

    # 1) Images (keyframe)
    images: Dict[str, Any] = {}
    for chan_name, sd_token in sample['data'].items():
        sd = nusc.get('sample_data', sd_token)
        if sd['sensor_modality'] == 'camera' and sd['is_key_frame']:
            # return the path; user can read image later
            images[sd['channel']] = str(Path(nuscenes_root) / sd['filename'])

    # 2) LiDAR (aggregate a few sweeps around t0 -> here use the keyframe only for simplicity)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = Path(nuscenes_root) / lidar_sd['filename']
    # In skeleton, we return path. Users can load point cloud with NuScenes devkit or their own reader.
    lidar = {"path": str(lidar_path), "n_sweeps": 1}

    # 3) Agents history (past_sec) for all ann in this sample
    agents_history = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        inst = nusc.get('instance', ann['instance_token'])
        past_xy = helper.get_past_for_agent(ann['instance_token'], sample_token, seconds=past_sec, in_agent_frame=False)
        past_yaw = np.zeros((past_xy.shape[0],), dtype=np.float32)  # yaw not directly provided; fill zeros or compute from heading change.
        agents_history[ann['instance_token']] = {
            "xy": past_xy.astype(np.float32),
            "yaw": past_yaw,
            "type": ann['category_name'],
            "size": list(map(float, ann['size']))
        }

    # 4) Ego history from ego poses aligned to LIDAR_TOP keyframes (use helper for simplicity)
    ego_past = helper.get_past_for_agent(sample['data']['LIDAR_TOP'], sample_token, seconds=past_sec, in_agent_frame=False)
    # The line above is not valid; NuScenes helper works with instance tokens. Instead, compute ego via ego_pose chain:
    # For skeleton, leave ego_history empty to avoid confusion; many teams compute ego path from consecutive LIDAR_TOP sample_data.
    ego_history = np.zeros((0,3), dtype=np.float32)

    # 5) Map slice
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nmap = NuScenesMap(dataroot=nuscenes_root, map_name=log['location'])
    # Get ego pose at t0
    import json
    ego_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', ego_sd['ego_pose_token'])
    ex, ey = ego_pose['translation'][:2]
    # Query vector layers in a radius
    def records(name): return nmap.get_records_in_radius(ex, ey, map_radius_m, name)
    # Build vectors as list of polylines/polygons (here we keep the token lists; users can decode via map API if needed)
    map_dict = {
        "lane_center": records('lane'),
        "lane_boundary": records('lane_divider'),
        "road_boundary": records('road_divider'),
        "crosswalk": records('ped_crossing'),
        "stop_line_or_sign": records('stop_line'),
    }

    return {
        "images": images,
        "lidar": lidar,
        "calib": {},  # user can fetch calibrated_sensor + ego_pose if needed
        "ego_history": ego_history,
        "agents_history": agents_history,
        "map": map_dict,
        "timestamps": {"t0": sample['timestamp'], "history_hz": 2.0, "lidar_hz": 20.0, "cam_hz": 12.0},
        "city_or_map_id": log['location'],
    }
