from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

def _optional_import():
    try:
        from av2.map.map_api import ArgoverseStaticMap
        return ArgoverseStaticMap
    except Exception as e:
        raise ImportError("`av2` package is required for Argoverse 2 Sensor loader. Install `av2`.") from e

def load_log(av2_log_dir: str, choose_t0: str = "mid") -> Dict[str, Any]:
    """Return a unified sample dict for one Argoverse 2 Sensor log.
    choose_t0: 'mid' or timestamp_ns string to choose a LiDAR sweep as t0.
    """
    ArgoverseStaticMap = _optional_import()
    log_dir = Path(av2_log_dir)

    # 1) pick a LiDAR sweep as t0
    sweeps = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
    if not sweeps:
        raise FileNotFoundError("No LiDAR feather files found under sensors/lidar")
    t0_file = sweeps[len(sweeps)//2] if choose_t0 == "mid" else (log_dir / "sensors" / "lidar" / f"{choose_t0}.feather")
    t0_ns = int(t0_file.stem)

    # 2) LiDAR points
    pc = pd.read_feather(t0_file).to_numpy()
    # columns: x,y,z,intensity,laser_number,offset_ns
    lidar_xyzi = pc[:, :4].astype(np.float32)

    # 3) One camera image path nearest to t0 (ring_front_center as example)
    cam_dir = log_dir / "sensors" / "cameras" / "ring_front_center" / "images"
    cam_files = sorted(cam_dir.glob("*.jpg"))
    img_path = str(cam_files[len(cam_files)//2]) if cam_files else ""

    # 4) Agents history from annotations.feather (past ~4s)
    ann = pd.read_feather(log_dir / "annotations.feather")
    H_ns = 4 * 1_000_000_000
    hist = ann[(ann.timestamp_ns <= t0_ns) & (ann.timestamp_ns >= t0_ns - H_ns)]
    agents_history = {}
    for tid, df in hist.groupby("track_uuid"):
        xy = df[["tx_m","ty_m"]].to_numpy(dtype=np.float32)
        sz = df[["length_m","width_m","height_m"]].iloc[-1].to_numpy(dtype=np.float32).tolist()
        agents_history[str(tid)] = {"xy": xy, "yaw": np.zeros((xy.shape[0],), dtype=np.float32),
                                    "type": str(df["category"].iloc[-1]), "size": sz}

    # 5) Ego history (use ego poses)
    ego_pose = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    # Reduce to last ~4s
    ego_hist = ego_pose[ego_pose["timestamp_ns"].between(t0_ns - H_ns, t0_ns)].copy()
    ego_xy_yaw = ego_hist[["tx_m","ty_m","yaw_rad"]].to_numpy(dtype=np.float32) if "yaw_rad" in ego_hist.columns else np.zeros((0,3), dtype=np.float32)

    # 6) Map
    avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=False)
    map_dict = {
        "lane_center": list(avm.vector_lane_segment_dict.keys()),
        "lane_boundary": [],
        "road_boundary": [],
        "crosswalk": list(avm.ped_crossing_dict.keys()),
        "stop_line_or_sign": list(avm.stop_line_dict.keys()) if hasattr(avm, "stop_line_dict") else [],
    }

    return {
        "images": {"ring_front_center": img_path},
        "lidar": lidar_xyzi,
        "calib": {},   # read intrinsics/extrinsics from calibration/ if needed
        "ego_history": ego_xy_yaw,
        "agents_history": agents_history,
        "map": map_dict,
        "timestamps": {"t0": t0_ns, "history_hz": 20.0, "lidar_hz": 10.0, "cam_hz": 20.0},
        "city_or_map_id": (log_dir / "map").name,
    }
