# datasets/packed_nuscenes_dataset.py
import json, numpy as np, torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple

def _squeeze_last(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return np.squeeze(a, axis=-1) if (a.ndim >= 1 and a.shape[-1] == 1) else a

def _np_to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    return torch.tensor(x, dtype=dtype)

def _pad_agents_to(a_xy_h, a_yaw_h, a_m_h,
                   a_xy_f, a_yaw_f, a_m_f,
                   target_Na: int) -> Tuple[np.ndarray, ...]:
    """
    Pad agent arrays to target_Na by appending all-NaN (mask=False) rows.
    Shapes:
      xy_h: (Na, Th, 2), yaw_h: (Na, Th), m_h: (Na, Th)
      xy_f: (Na, Tf, 2), yaw_f: (Na, Tf), m_f: (Na, Tf)
    """
    Na, Th, _ = a_xy_h.shape
    Tf = a_xy_f.shape[1]
    if Na == target_Na:
        return a_xy_h, a_yaw_h, a_m_h, a_xy_f, a_yaw_f, a_m_f

    pad_N = target_Na - Na
    nan_h_xy = np.full((pad_N, Th, 2), np.nan, np.float32)
    nan_h_yaw = np.full((pad_N, Th), np.nan, np.float32)
    false_h = np.zeros((pad_N, Th), bool)

    nan_f_xy = np.full((pad_N, Tf, 2), np.nan, np.float32)
    nan_f_yaw = np.full((pad_N, Tf), np.nan, np.float32)
    false_f = np.zeros((pad_N, Tf), bool)

    a_xy_h = np.concatenate([a_xy_h, nan_h_xy], 0)
    a_yaw_h = np.concatenate([a_yaw_h, nan_h_yaw], 0)
    a_m_h = np.concatenate([a_m_h, false_h], 0)

    a_xy_f = np.concatenate([a_xy_f, nan_f_xy], 0)
    a_yaw_f = np.concatenate([a_yaw_f, nan_f_yaw], 0)
    a_m_f = np.concatenate([a_m_f, false_f], 0)
    return a_xy_h, a_yaw_h, a_m_h, a_xy_f, a_yaw_f, a_m_f

class PackedNuScenesDataset(Dataset):
    """
    Dataset over manifest.jsonl rows. Each item returns tensors ready for training.
    History/Future 均為「左對齊有效前綴 + 右側補空」(mask=True 在前段、尾端 False)。
    """
    def __init__(self, manifest_path: str, include_map: bool = False):
        self.rows: List[Dict[str, Any]] = [
            json.loads(l) for l in open(manifest_path, "r", encoding="utf-8") if l.strip()
        ]
        if len(self.rows) == 0:
            raise ValueError("Empty manifest.")
        self.include_map = include_map

        # infer Th/Tf from first row's file (consistency check)
        d0 = np.load(self.rows[0]["npz"], allow_pickle=True)
        self.Th = int(d0["ego_hist_xy"].shape[0])
        self.Tf = int(d0["ego_fut_xy"].shape[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.rows[i]
        d = np.load(row["npz"], allow_pickle=True)

        # ego
        ego_hist_xy  = np.asarray(d["ego_hist_xy"], dtype=np.float32)          # (Th,2)
        ego_hist_yaw = _squeeze_last(d["ego_hist_yaw"]).astype(np.float32)     # (Th,)
        ego_hist_mask= np.asarray(d["ego_hist_mask"]).astype(bool)             # (Th,)

        ego_fut_xy   = np.asarray(d["ego_fut_xy"], dtype=np.float32)           # (Tf,2)
        ego_fut_yaw  = _squeeze_last(d["ego_fut_yaw"]).astype(np.float32)      # (Tf,)
        ego_fut_mask = np.asarray(d["ego_fut_mask"]).astype(bool)              # (Tf,)

        # agents
        A_xy_h  = np.asarray(d["agents_hist_xy"], dtype=np.float32)            # (Na,Th,2)
        A_yaw_h = _squeeze_last(d["agents_hist_yaw"]).astype(np.float32)       # (Na,Th)
        A_m_h   = np.asarray(d["agents_hist_mask"]).astype(bool)               # (Na,Th)

        A_xy_f  = np.asarray(d["agents_fut_xy"], dtype=np.float32)             # (Na,Tf,2)
        A_yaw_f = _squeeze_last(d["agents_fut_yaw"]).astype(np.float32)        # (Na,Tf)
        A_m_f   = np.asarray(d["agents_fut_mask"]).astype(bool)                # (Na,Tf)

        Th, Tf = ego_hist_xy.shape[0], ego_fut_xy.shape[0]
        assert Th == self.Th and Tf == self.Tf, "Inconsistent Th/Tf across files."

        out = {
            # ego
            "ego_hist_xy":  _np_to_torch(ego_hist_xy),
            "ego_hist_yaw": _np_to_torch(ego_hist_yaw),
            "ego_hist_mask": torch.from_numpy(ego_hist_mask),
            "ego_fut_xy":   _np_to_torch(ego_fut_xy),
            "ego_fut_yaw":  _np_to_torch(ego_fut_yaw),
            "ego_fut_mask": torch.from_numpy(ego_fut_mask),

            # agents (先保留 numpy，collate 時統一補齊/stack)
            "_A_xy_h":  A_xy_h, "_A_yaw_h": A_yaw_h, "_A_m_h": A_m_h,
            "_A_xy_f":  A_xy_f, "_A_yaw_f": A_yaw_f, "_A_m_f": A_m_f,

            # meta
            "idx": int(row["idx"]),
            "npz": row["npz"],
            "sample_token": row["sample_token"],
            "scene_token": row["scene_token"],
            "location": row["location"],
        }

        if self.include_map:
            out["map_lane_center"] = d["map_lane_center"]
            out["map_lane_divider"] = d["map_lane_divider"]
            out["map_road_divider"] = d["map_road_divider"]
            out["map_ped_crossing"] = d["map_ped_crossing"]
            out["map_stop_line"] = d["map_stop_line"]
            out["map_traffic_light"] = d["map_traffic_light"]

        return out

def packed_nuscenes_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad agents to max_Na in the batch; stack everything into tensors.
    """
    Th = batch[0]["ego_hist_xy"].shape[0]
    Tf = batch[0]["ego_fut_xy"].shape[0]
    # max Na within the batch
    Nas = [b["_A_xy_h"].shape[0] for b in batch]
    max_Na = max(Nas) if Nas else 0

    # stack ego
    ego_hist_xy   = torch.stack([b["ego_hist_xy"] for b in batch], 0)      # (B,Th,2)
    ego_hist_yaw  = torch.stack([b["ego_hist_yaw"] for b in batch], 0)     # (B,Th)
    ego_hist_mask = torch.stack([b["ego_hist_mask"] for b in batch], 0)    # (B,Th)
    ego_fut_xy    = torch.stack([b["ego_fut_xy"] for b in batch], 0)       # (B,Tf,2)
    ego_fut_yaw   = torch.stack([b["ego_fut_yaw"] for b in batch], 0)      # (B,Tf)
    ego_fut_mask  = torch.stack([b["ego_fut_mask"] for b in batch], 0)     # (B,Tf)

    # agents: pad to max_Na then stack
    A_xy_h_list, A_yaw_h_list, A_m_h_list = [], [], []
    A_xy_f_list, A_yaw_f_list, A_m_f_list = [], [], []

    for b in batch:
        A_xy_h, A_yaw_h, A_m_h = b["_A_xy_h"], b["_A_yaw_h"], b["_A_m_h"]
        A_xy_f, A_yaw_f, A_m_f = b["_A_xy_f"], b["_A_yaw_f"], b["_A_m_f"]
        A_xy_h, A_yaw_h, A_m_h, A_xy_f, A_yaw_f, A_m_f = _pad_agents_to(
            A_xy_h, A_yaw_h, A_m_h, A_xy_f, A_yaw_f, A_m_f, max_Na
        )
        A_xy_h_list.append(torch.from_numpy(A_xy_h).float())
        A_yaw_h_list.append(torch.from_numpy(A_yaw_h).float())
        A_m_h_list.append(torch.from_numpy(A_m_h))
        A_xy_f_list.append(torch.from_numpy(A_xy_f).float())
        A_yaw_f_list.append(torch.from_numpy(A_yaw_f).float())
        A_m_f_list.append(torch.from_numpy(A_m_f))

    if max_Na > 0:
        agents_hist_xy   = torch.stack(A_xy_h_list, 0)   # (B,Na,Th,2)
        agents_hist_yaw  = torch.stack(A_yaw_h_list,0)   # (B,Na,Th)
        agents_hist_mask = torch.stack(A_m_h_list,  0)   # (B,Na,Th) bool
        agents_fut_xy    = torch.stack(A_xy_f_list, 0)   # (B,Na,Tf,2)
        agents_fut_yaw   = torch.stack(A_yaw_f_list,0)   # (B,Na,Tf)
        agents_fut_mask  = torch.stack(A_m_f_list,  0)   # (B,Na,Tf) bool
    else:
        agents_hist_xy = agents_hist_yaw = agents_hist_mask = None
        agents_fut_xy = agents_fut_yaw = agents_fut_mask = None

    out = {
        "ego_hist_xy": ego_hist_xy, "ego_hist_yaw": ego_hist_yaw, "ego_hist_mask": ego_hist_mask,
        "ego_fut_xy":  ego_fut_xy,  "ego_fut_yaw":  ego_fut_yaw,  "ego_fut_mask":  ego_fut_mask,
        "agents_hist_xy": agents_hist_xy, "agents_hist_yaw": agents_hist_yaw, "agents_hist_mask": agents_hist_mask,
        "agents_fut_xy":  agents_fut_xy,  "agents_fut_yaw":  agents_fut_yaw,  "agents_fut_mask":  agents_fut_mask,
        "meta": {
            "idx": [b["idx"] for b in batch],
            "sample_token": [b["sample_token"] for b in batch],
            "scene_token": [b["scene_token"] for b in batch],
            "npz": [b["npz"] for b in batch],
            "location": [b["location"] for b in batch],
        }
    }
    return out
