from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

def _optional_import():
    try:
        from nuscenes import NuScenes
        from nuscenes.prediction import PredictHelper
        from nuscenes.map_expansion.map_api import NuScenesMap
        return NuScenes, PredictHelper, NuScenesMap
    except Exception as e:
        raise ImportError("nuscenes-devkit is required for this loader. Install `nuscenes-devkit`.") from e

def _history_tokens(nusc, t0_sample_token: str, seconds: float) -> List[str]:
    """Collect sample_tokens from oldest -> t0 within the past `seconds` window."""
    s0 = nusc.get('sample', t0_sample_token)
    tokens = [t0_sample_token]; t0 = s0['timestamp']; cur = s0
    max_steps = int(np.ceil(seconds * 2.5)) + 1  # 2Hz keyframes, +1 to include t0
    while cur.get('prev') and len(tokens) < max_steps:
        prev_tok = cur['prev']; prev_s = nusc.get('sample', prev_tok)
        if (t0 - prev_s['timestamp'])/1e6 > seconds: break
        tokens.append(prev_tok); cur = prev_s
    tokens = tokens[::-1]  # oldest -> t0
    return tokens

def _instance_ann_map(nusc, instance_token: str) -> Dict[str, dict]:
    """Build a dict: sample_token -> sample_annotation for a given instance."""
    inst = nusc.get('instance', instance_token)
    ann_token = inst['first_annotation_token']
    m = {}
    while ann_token:
        ann = nusc.get('sample_annotation', ann_token)
        m[ann['sample_token']] = ann
        ann_token = ann['next'] if ann['next'] else None
    return m

def _get_agents_histories_aligned(nusc, instance_tokens: List[str], t0_sample_token: str, seconds: float):
    """
    For many agents at once, return histories aligned on the SAME time grid.
    Returns:
      tokens: [T] sample_tokens (oldest->t0)
      out: dict[instance_token] = {
          "xy":   [T,2] float32  (NaN where missing),
          "yaw":  [T]   float32  (NaN where missing),
          "mask": [T]   bool     (True where valid)
      }
    """
    tokens = _history_tokens(nusc, t0_sample_token, seconds)
    T = len(tokens)
    out = {}
    for inst in instance_tokens:
        annmap = _instance_ann_map(nusc, inst)
        xy   = np.full((T, 2), 0, dtype=np.float32)
        yaw  = np.full((T,),   0, dtype=np.float32)
        mask = np.zeros((T,),   dtype=bool)

        for i, tok in enumerate(tokens):
            ann = annmap.get(tok)
            if ann is None:
                continue
            x, y, _ = ann['translation']
            qw, qx, qy, qz = ann['rotation']   # [w,x,y,z]
            th = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            xy[i] = [x, y]; yaw[i] = th; mask[i] = True

        out[inst] = {"xy": xy, "yaw": yaw, "mask": mask}
    return tokens, out

def _get_agent_history(nusc, instance_token: str, t0_sample_token: str, seconds: float = 4.0) -> np.ndarray:
    """
    Return agent history as [[x, y, yaw], ...] in WORLD frame, oldest->t0.
    Yaw is derived from sample_annotation['rotation'] quaternion (w,x,y,z), same as ego.
    """
    # 建立 [最舊 ... t0] 的 sample_token 序列（跟 _get_ego_history 同步邏輯）
    s0 = nusc.get('sample', t0_sample_token)
    tokens = [t0_sample_token]
    t0 = s0['timestamp']
    cur = s0
    max_steps = int(np.ceil(seconds * 2.5)) + 1
    while cur.get('prev') and len(tokens) < max_steps:
        prev_tok = cur['prev']
        prev_s = nusc.get('sample', prev_tok)
        if (t0 - prev_s['timestamp'])/1e6 > seconds: break
        tokens.append(prev_tok); cur = prev_s
    tokens = tokens[::-1]  # oldest -> t0

    # 把這個 instance 的整條 annotation 鏈做成 dict：sample_token -> annotation
    inst = nusc.get('instance', instance_token)
    ann_token = inst['first_annotation_token']
    samp2ann = {}
    while ann_token:
        ann = nusc.get('sample_annotation', ann_token)
        samp2ann[ann['sample_token']] = ann
        ann_token = ann['next'] if ann['next'] else None

    xy_yaw = []
    for tok in tokens:
        ann = samp2ann.get(tok)
        if ann is None:
            # 這個時間點沒有這個 agent（可能進場前/出場後），略過
            continue
        x, y, _ = ann['translation']
        qw, qx, qy, qz = ann['rotation']  # nuScenes 四元數格式: [w, x, y, z]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        xy_yaw.append([x, y, yaw])

    return np.array(xy_yaw, dtype=np.float32)

def _get_ego_history(
    nusc,
    sample_token: str,
    seconds: float = 4.0,
    tokens: List[str] = None
) -> np.ndarray:
    """
    Return ego history as [[x, y, yaw], ...] in WORLD frame, oldest->t0.
    If `tokens` is provided, it must be the time grid (oldest->t0) shared with agents.
    """
    if tokens is None:
        tokens = _history_tokens(nusc, sample_token, seconds)

    xy_yaw = []
    for tok in tokens:
        s = nusc.get('sample', tok)
        lidar_sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        x, y, _ = ego_pose['translation']
        qw, qx, qy, qz = ego_pose['rotation']  # [w,x,y,z]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        xy_yaw.append([x, y, yaw])

    return np.array(xy_yaw, dtype=np.float32)

'''
def _get_ego_history(nusc, sample_token: str, seconds: float = 4.0) -> np.ndarray:
    sample = nusc.get('sample', sample_token)
    tokens = [sample_token]; t0 = sample['timestamp']; cur = sample
    max_steps = int(np.ceil(seconds * 2.5)) + 1
    while cur.get('prev') and len(tokens) < max_steps:
        prev_tok = cur['prev']; prev_s = nusc.get('sample', prev_tok)
        if (t0 - prev_s['timestamp'])/1e6 > seconds: break
        tokens.append(prev_tok); cur = prev_s
    tokens = tokens[::-1]
    xy_yaw = []
    for tok in tokens:
        s = nusc.get('sample', tok)
        lidar_sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        x, y, _ = ego_pose['translation']
        qw, qx, qy, qz = ego_pose['rotation']
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        xy_yaw.append([x, y, yaw])
    return np.array(xy_yaw, dtype=np.float32)
'''

def _map_slice_tokens(nmap, ex: float, ey: float, radius: float) -> Dict[str, List[str]]:
    def rec(layer):
        try: return nmap.get_records_in_radius(ex, ey, radius, layer)
        except Exception: return []
    return {
        "lane_center": rec('lane'),
        "lane_boundary": rec('lane_divider'),
        "road_boundary": rec('road_divider'),
        "crosswalk": rec('ped_crossing'),
        "stop_line_or_sign": rec('stop_line'),
    }

def load_sample(nuscenes_root: str, sample_token: str, past_sec: float = 4.0, map_radius_m: float = 80.0) -> Dict[str, Any]:
    """Return a unified sample dict for one nuScenes keyframe sample."""
    NuScenes, PredictHelper, NuScenesMap = _optional_import()
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root)
    helper = PredictHelper(nusc)
    sample = nusc.get('sample', sample_token)

    # 1) image 路徑（keyframe）
    images: Dict[str, Any] = {}
    for _, sd_token in sample['data'].items():
        sd = nusc.get('sample_data', sd_token)
        if sd['sensor_modality'] == 'camera' and sd['is_key_frame']:
            # return the path; can read image later
            images[sd['channel']] = str(Path(nuscenes_root) / sd['filename'])

    # 2) LiDAR：回傳檔案路徑 + meta
    # In skeleton, we return path. Users can load point cloud with NuScenes devkit or their own reader.
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar = {"path": str(Path(nuscenes_root) / lidar_sd['filename']), "n_sweeps": 1}

    # 3) Agents history (x, y, yaw) for all ann in this sample 多代理歷史
    # 先收集 t0 畫面上的 instances
    inst_at_t0 = []
    inst_meta  = {}  # type/size 取自 t0 時刻的 ann
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        inst_at_t0.append(ann['instance_token'])
        inst_meta[ann['instance_token']] = {
            "type": ann['category_name'],
            "size": list(map(float, ann['size'])),
        }

    tokens_hist, aligned = _get_agents_histories_aligned(
        nusc, inst_at_t0, sample_token, seconds=past_sec
    )

    agents_history = {}
    for inst in inst_at_t0:
        ah = aligned.get(inst, None)
        if ah is None:
            continue
        meta = inst_meta[inst]
        agents_history[inst] = {
            "xy":   ah["xy"],            # [T,2] 世界座標（oldest->t0），缺值為 NaN
            "yaw":  ah["yaw"],           # [T]   世界朝向，缺值為 NaN
            "mask": ah["mask"],          # [T]   是否有效
            "type": meta["type"],
            "size": meta["size"],
        }

    '''
    agents_history = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        inst_tok = ann['instance_token']

        # 用四元數（與 ego 相同做法）取回 [x,y,yaw]，時間順序為 oldest->t0
        xy_yaw = _get_agent_history(nusc, inst_tok, sample_token, seconds=past_sec)

        # 若這個 agent 在該視窗內完全不存在就略過（可選：也可保留空陣列）
        if xy_yaw.size == 0:
            continue

        agents_history[inst_tok] = {
            "xy":  xy_yaw[:, :2].astype(np.float32),
            "yaw": xy_yaw[:, 2].astype(np.float32),
            "type": ann['category_name'],
            "size": list(map(float, ann['size'])),
        }
    '''

    '''
    agents_history = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        past_xy = helper.get_past_for_agent(ann['instance_token'], sample_token, seconds=past_sec, in_agent_frame=False)
        past_yaw = np.zeros((past_xy.shape[0],), dtype=np.float32)  # yaw not directly provided; fill zeros or compute from heading change.
        agents_history[ann['instance_token']] = {
            "xy": past_xy.astype(np.float32),
            "yaw": past_yaw,
            "type": ann['category_name'],
            "size": list(map(float, ann['size']))
        }
    '''

    # 4) ego 歷史
    #ego_history = _get_ego_history(nusc, sample_token, seconds=past_sec)
    ego_history = _get_ego_history(nusc, sample_token, seconds=past_sec, tokens=tokens_hist)

    # 5) 地圖切片（回 token；需要座標可用 map API 再展開）
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nmap = NuScenesMap(dataroot=nuscenes_root, map_name=log['location'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ex, ey = ego_pose['translation'][:2]
    map_dict = _map_slice_tokens(nmap, ex, ey, map_radius_m)

    # 6) 取 calib and t0 參考（用 LIDAR_TOP 的 ego_pose）
    calib_sensor_t0 = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    R_el = Quaternion(calib_sensor_t0["rotation"]).rotation_matrix
    t_el = np.array(calib_sensor_t0["translation"], dtype=np.float32)
    pose_t0  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    R_we = Quaternion(pose_t0['rotation']).rotation_matrix
    t_we = np.array(pose_t0['translation'], dtype=np.float32)

    calib = {
        "world_from_ego": {"R": R_we.tolist(), "t": t_we.tolist()},
        "sensors": {}
    }

    for chan, sd_token in sample['data'].items():
        sd  = nusc.get('sample_data', sd_token)
        cal = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        R_es = Quaternion(cal['rotation']).rotation_matrix
        t_es = np.array(cal['translation'], dtype=np.float32)

        entry = {
            "ego_from_sensor": {"R": R_es.tolist(), "t": t_es.tolist()},
            "timestamp": sd["timestamp"],
            "sample_data_token": sd_token,
        }
        if sd['sensor_modality'] == 'camera':
            entry["K"] = cal["camera_intrinsic"]          # 3x3
            entry["img_size"] = [sd["height"], sd["width"]]
            entry["distortion"] = None                    # nuScenes 未提供畸變
        calib["sensors"][chan] = entry

    return {
        "images": images,
        "lidar": lidar,
        "calib": calib,
        "ego_history": ego_history,
        "history_tokens": tokens_hist,
        "agents_history": agents_history,
        "map": map_dict,
        "timestamps": {"t0": sample['timestamp'], "history_hz": 2.0, "lidar_hz": 20.0, "cam_hz": 12.0},
        "city_or_map_id": log['location'],
        "t0": {
                "lidar_filename": str(Path(nuscenes_root) / lidar_sd['filename']),
                "calib_rot": R_el,
                "calib_trans": t_el,
                "ego_rot": R_we,
                "ego_trans": t_we,
        }
    }
