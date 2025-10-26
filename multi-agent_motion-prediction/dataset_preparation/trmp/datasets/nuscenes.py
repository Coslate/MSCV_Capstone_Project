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

def _future_tokens(nusc, t0_sample_token: str, seconds: float) -> List[str]:
    """Collect sample_tokens from t1 -> t{+} within the next `seconds` window (exclude t0)."""
    s0 = nusc.get('sample', t0_sample_token)
    t0 = s0['timestamp']
    tokens = []
    cur = s0
    max_steps = int(np.ceil(seconds * 2.0))  # 2Hz keyframes
    while cur.get('next') and len(tokens) < max_steps:
        nxt_tok = cur['next']; nxt_s = nusc.get('sample', nxt_tok)
        if (nxt_s['timestamp'] - t0)/1e6 > seconds: break
        tokens.append(nxt_tok); cur = nxt_s
    return tokens  # ascending (t1 -> ...)

def _get_agents_futures_aligned(nusc, instance_tokens: List[str], t0_sample_token: str, seconds: float):
    """
    Like _get_agents_histories_aligned, but for FUTURE (exclude t0).
    Returns:
      tokens: [Tf] (t1->t{+})
      out: dict[instance_token] = {'xy':[Tf,2], 'yaw':[Tf], 'mask':[Tf]}
    """
    tokens = _future_tokens(nusc, t0_sample_token, seconds)
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
            qw, qx, qy, qz = ann['rotation']
            th = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            xy[i] = [x, y]; yaw[i] = th; mask[i] = True
        out[inst] = {"xy": xy, "yaw": yaw, "mask": mask}
    return tokens, out

def _get_ego_future(
    nusc,
    sample_token: str,
    seconds: float = 6.0,
    tokens: List[str] = None
) -> np.ndarray:
    """
    Return ego future as [[x, y, yaw], ...] in WORLD frame, t1->t{+}.
    If `tokens` given, must be the FUTURE grid returned by _future_tokens.
    """
    if tokens is None:
        tokens = _future_tokens(nusc, sample_token, seconds)
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

def _map_slice_tokens(nmap, ex: float, ey: float, radius: float) -> Dict[str, List[str]]:
    """Return tokens around (ex, ey) within radius for unified map layers."""
    ex = float(ex); ey = float(ey); radius = float(radius)

    def rec(layer: str) -> List[str]:
        try:
            # get_records_in_radius 期待 list[str]
            out = nmap.get_records_in_radius(ex, ey, radius, [layer])
            return out.get(layer, [])
        except Exception as e:
            print(f"[HDMap] get_records_in_radius(layer='{layer}') failed: {e}")
            return []

    # 中心線同時包含 lane 與 lane_connector（Option A）
    lane = rec("lane")
    lane_conn = rec("lane_connector")
    lane_tokens = lane+lane_conn

    return {
        "lane_center":   lane_tokens,       # 合併給畫中心線用
        "lane_connector": lane_conn,        # 額外保留，方便統計/調試/特殊樣式
        "lane_divider":  rec("lane_divider"),
        "road_divider":  rec("road_divider"),
        "ped_crossing":  rec("ped_crossing"),
        "stop_line":     rec("stop_line"),
        "traffic_light": rec("traffic_light"),
    }

def load_sample(
    nuscenes_root: str,
    sample_token: str,
    past_sec: float = 4.0,
    future_sec: float = 6.0,
    map_radius_m: float = 80.0,
    *,
    version: str = "v1.0-trainval",
    nusc=None,                 # <-- 新增：可傳入共用 NuScenes 物件
    nmap_cache: dict = None    # <-- 新增：共用 NuScenesMap cache (location)
) -> Dict[str, Any]:    
    """Return a unified sample dict for one nuScenes keyframe sample."""
    NuScenes, PredictHelper, NuScenesMap = _optional_import()
    if nusc is None:
        nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)

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

    # 4a) ego history (x: data)
    ego_history = _get_ego_history(nusc, sample_token, seconds=past_sec, tokens=tokens_hist)

    # 4b) ego future（y: labels）：以 t0 畫面上的 instances 對齊取未來 Tf 步（exclude t0）
    tokens_fut, aligned_fut = _get_agents_futures_aligned(
        nusc, inst_at_t0, sample_token, seconds=future_sec
    )

    agents_future = {}
    for inst in inst_at_t0:
        af = aligned_fut.get(inst, None)
        if af is None:
            continue
        meta = inst_meta[inst]
        agents_future[inst] = {
            "xy":   af["xy"],     # [Tf,2] world
            "yaw":  af["yaw"],    # [Tf]
            "mask": af["mask"],   # [Tf]
            "type": meta["type"],
            "size": meta["size"],
        }

    ego_future = _get_ego_future(nusc, sample_token, seconds=future_sec, tokens=tokens_fut)


    # 5) 地圖切片（回 token；需要座標可用 map API 再展開）
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']

    if nmap_cache is not None:
        if map_name not in nmap_cache:
            nmap_cache[map_name] = NuScenesMap(dataroot=nuscenes_root, map_name=map_name)
        nmap = nmap_cache[map_name]
    else:
        # 沒提供 cache 就自己建（單次）
        nmap = NuScenesMap(dataroot=nuscenes_root, map_name=map_name)

    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ex, ey = ego_pose['translation'][:2]
    map_dict = _map_slice_tokens(nmap, ex, ey, map_radius_m)

    # 6) 取 calib and t0 參考（用 LIDAR_TOP 的 ego_pose）
    calib_sensor_t0 = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    R_el = Quaternion(calib_sensor_t0["rotation"]).rotation_matrix #ego <- lidar
    t_el = np.array(calib_sensor_t0["translation"], dtype=np.float32)
    pose_t0  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])    #world <- ego
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
            "channel": sd["channel"],
            "modality": sd["sensor_modality"],      # 'camera' | 'lidar' | 'radar'
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
        "future_tokens": tokens_fut,
        "ego_future": ego_future,
        "agents_future": agents_future,
        "map": map_dict,
        "timestamps": {"t0": sample['timestamp'], "history_hz": 2.0, "lidar_hz": 20.0, "cam_hz": 12.0},
        "city_or_map_id": map_name,
        "t0": {
                "lidar_filename": str(Path(nuscenes_root) / lidar_sd['filename']),
                "calib_rot": R_el,
                "calib_trans": t_el,
                "ego_rot": R_we,
                "ego_trans": t_we,
        }
    }
