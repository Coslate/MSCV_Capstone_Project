# Traffic-Rule-Aware Motion Prediction — Data Skeleton

This repo gives you a **unified dataloader interface** for three datasets where **ego camera + LiDAR**, **HD map**, and **multi-agent history** are time-aligned within the same scene/log:

- **nuScenes v1.0**
- **Argoverse 2 — Sensor Dataset**
- **Waymo Open Dataset — Perception (with maps)**

> Sound modality is intentionally **not used**.

## What you get
- A single **unified sample format** (Python dict) your model can consume.
- Minimal loaders for each dataset (`trmp/datasets/*.py`), returning the unified format.
- A simple `DataModule` wrapper to iterate batches.
- A basic **BEV visualizer** to verify synchronization (map + agents + LiDAR overlay).
- Example **configs** and **demo scripts** for each dataset.

## Install (minimal)
```bash
# create a venv first if you like
pip install -r requirements.txt
```

> Notes:
> - nuScenes is the easiest to test end-to-end with this skeleton.
> - Argoverse 2 & Waymo imports are optional; the corresponding loader will soft-fail with a helpful message if the package is missing on your system.
> - Waymo Perception + maps support requires the TFRecord reader & the map-enabled v1.4.2/1.4.3 format (not the modular v2.0 map-less format).

## Unified Output Interface
Each sample from a loader returns a `dict`:
```python
sample = {
  "images": {cam_name: np.ndarray[H,W,3] or str_path},
  "lidar":  np.ndarray[N, 4],  # (x,y,z,intensity) in a common frame
  "calib":  {...},             # intrinsics/extrinsics, ego pose at t0
  "ego_history": np.ndarray[T, 3],  # x,y,yaw (or x,y,z,yaw if configured)
  "agents_history": {
      agent_id: {"xy": np.ndarray[T,2], "yaw": np.ndarray[T], "type": str, "size": [l,w,h]}
  },
  "map": {
    "lane_center": [np.ndarray[K_i,2]],
    "lane_boundary": [np.ndarray[...]],
    "road_boundary": [np.ndarray[...]],
    "crosswalk": [np.ndarray[...]],
    "stop_line_or_sign": [np.ndarray[...]],  # stop_line (nuScenes/AV2) or stop_sign poly (Waymo)
  },
  "timestamps": {"t0": int, "history_hz": float, "lidar_hz": float, "cam_hz": float},
  "city_or_map_id": str,
}
```

## Run a quick sanity check (nuScenes example)
```bash
python scripts/demo_nuscenes.py --dataroot /path/to/nuscenes --sample_token <some_sample_token>     --out_png /tmp/preview.png
```

This will load one sample, aggregate LiDAR around t0, gather agent histories (~4s), slice a local HD map window, and render a BEV PNG for quick inspection.

## Requirements (minimal, version hints)
See `requirements.txt`. If you don't plan to use a specific dataset, you can omit its dependency.
- nuScenes: `nuscenes-devkit`
- Argoverse 2: `av2`
- Waymo Perception: `tensorflow==2.11.*` + `waymo-open-dataset-tf-2-11-0` (or a matching build for your TF)
