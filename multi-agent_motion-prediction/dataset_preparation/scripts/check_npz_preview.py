#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 專案根目錄到 sys.path
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from trmp.utils.map_helper import visible

# ---------- tiny utils ----------
def wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi


def draw_heading(ax, x, y, theta,
                 shaft_len=5.0, head_len=2.5, head_w=1.5,
                 color=(0,0,0), alpha=0.8, z=5):
    """
    畫一支「箭桿長度 = shaft_len」的箭頭（總長 = shaft_len + head_len）。
    """
    L = float(shaft_len + head_len)  # 讓箭桿正好 = shaft_len
    ax.arrow(x, y,
             L*np.cos(theta), L*np.sin(theta),
             length_includes_head=True,
             head_width=head_w, head_length=head_len,
             fc=color, ec=color, alpha=alpha, zorder=z)

def _iter_polylines(obj) -> Iterable[np.ndarray]:
    """Yield Nx2 arrays from object arrays / lists / single arrays."""
    if obj is None: return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for it in obj:
            if it is None: continue
            arr = np.asarray(it)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                yield arr[:, :2].astype(np.float32, copy=False)
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            if it is None: continue
            arr = np.asarray(it)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                yield arr[:, :2].astype(np.float32, copy=False)
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
            yield arr[:, :2].astype(np.float32, copy=False)

def _iter_points(obj) -> np.ndarray:
    """Return Mx2 stacked points from object arrays / arrays."""
    if obj is None: return np.zeros((0,2), np.float32)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        pts = []
        for it in obj:
            if it is None: continue
            arr = np.asarray(it)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                pts.append(arr[:, :2])
        return np.vstack(pts) if pts else np.zeros((0,2), np.float32)
    arr = np.asarray(obj)
    return arr[:, :2] if (arr.ndim == 2 and arr.shape[1] >= 2) else np.zeros((0,2), np.float32)

def _color_for_key(key: str) -> Tuple[float,float,float]:
    return plt.get_cmap("tab20")(hash(str(key)) % 20)[:3]

def _boolean_mask_rows(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    if not (isinstance(a, np.ndarray) and isinstance(m, np.ndarray)): return a
    if m.dtype != bool or m.shape[0] != a.shape[0]: return a
    return a[m]

def load_manifest_row(manifest_path: str, idx: int=None, sample_token: str=None) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if idx is not None and int(row.get("idx",-1)) == int(idx): return row
            if sample_token is not None and row.get("sample_token") == sample_token: return row
    raise ValueError("Target not found in manifest.")

def summarize_npz(npz: Dict[str, Any]) -> str:
    keys = [
        "ego_hist_xy","ego_hist_yaw","ego_hist_mask",
        "ego_fut_xy","ego_fut_yaw","ego_fut_mask",
        "agents_hist_xy","agents_hist_yaw","agents_hist_mask",
        "agents_fut_xy","agents_fut_yaw","agents_fut_mask",
        "agents_type","agents_id",
        "map_lane_center","map_lane_divider","map_road_divider",
        "map_ped_crossing","map_stop_line","map_traffic_light",
        "cam_names","cam_paths","lidar_path",
        "location","sample_token","scene_token",
        "R_we","t_we","hist_hz","past_sec","future_sec","stride_sec","map_radius_m"
    ]
    lines = []
    for k in keys:
        if k in npz:
            v = npz[k]
            try: lines.append(f"{k:>18}: {tuple(v.shape)} {v.dtype}")
            except Exception: lines.append(f"{k:>18}: <no-shape>")
    return "\n".join(lines)

# ---------- LiDAR loader (lidar -> ego@t0) ----------
def load_lidar_points_ego(lidar_path: str, sample_token: str, dataroot: str, version: str):
    """
    Returns [N,4] (x,y,z,intensity) in ego@t0.
    Uses nuScenes to fetch ego<-lidar extrinsics at t0 for the given sample token.
    """
    try:
        from nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
    except Exception:
        return None
    if not os.path.exists(lidar_path): return None

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    s = nusc.get("sample", sample_token)
    sd_lidar = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    cal = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
    # ego_from_lidar: R_el, t_el
    from pyquaternion import Quaternion
    R_el = Quaternion(cal["rotation"]).rotation_matrix.astype(np.float32)
    t_el = np.array(cal["translation"], dtype=np.float32)

    pc = LidarPointCloud.from_file(lidar_path)  # [4,N]
    pts_lidar = pc.points[:3, :].T.astype(np.float32)  # [N,3]
    intensity = pc.points[3, :].astype(np.float32)     # [N]
    # row-vector: x_e = x_l @ R_el^T + t_el
    pts_ego = (pts_lidar @ R_el.T) + t_el
    return np.c_[pts_ego, intensity[:, None]]

# ---------- plot ----------
def plot_from_npz(npz_path: str, out_png: str, *, dataroot: str=None, version: str="v1.0-trainval", with_lidar: bool=True):
    data = np.load(npz_path, allow_pickle=True)

    # meta
    sample_token = data["sample_token"].item() if data["sample_token"].shape == () else str(data["sample_token"])
    location     = data["location"].item() if data["location"].shape == () else str(data["location"])
    r            = float(np.asarray(data["map_radius_m"]))
    hist_hz      = float(np.asarray(data["hist_hz"]))

    # ego (already in ego@t0 & yaw relative to t0)
    ego_xy_h  = np.asarray(data["ego_hist_xy"])
    ego_yaw_h = np.asarray(data["ego_hist_yaw"])
    ego_m_h   = np.asarray(data["ego_hist_mask"]).astype(bool)
    ego_xy_f  = np.asarray(data["ego_fut_xy"])
    ego_yaw_f = np.asarray(data["ego_fut_yaw"])
    ego_m_f   = np.asarray(data["ego_fut_mask"]).astype(bool)

    ego_xy_h_v  = _boolean_mask_rows(ego_xy_h, ego_m_h)
    ego_yaw_h_v = ego_yaw_h[ego_m_h] if ego_yaw_h.shape[0] == ego_m_h.shape[0] else ego_yaw_h
    ego_xy_f_v  = _boolean_mask_rows(ego_xy_f, ego_m_f)
    ego_yaw_f_v = ego_yaw_f[ego_m_f] if ego_yaw_f.shape[0] == ego_m_f.shape[0] else ego_yaw_f

    # agents (already in ego@t0 & yaw relative to t0)
    A_xy_h  = np.asarray(data["agents_hist_xy"])
    A_yaw_h = np.asarray(data["agents_hist_yaw"])
    A_m_h   = np.asarray(data["agents_hist_mask"]).astype(bool)
    A_xy_f  = np.asarray(data["agents_fut_xy"])
    A_yaw_f = np.asarray(data["agents_fut_yaw"])
    A_m_f   = np.asarray(data["agents_fut_mask"]).astype(bool)
    A_type  = np.asarray(data["agents_type"], dtype=object)
    A_id    = np.asarray(data["agents_id"],   dtype=object)
    Na = A_xy_h.shape[0]

    # map (already in ego@t0)
    map_lane_center = data["map_lane_center"]
    map_lane_div    = data["map_lane_divider"]
    map_road_div    = data["map_road_divider"]
    map_ped_xing    = data["map_ped_crossing"]
    map_stop_line   = data["map_stop_line"]
    map_tl_points   = data["map_traffic_light"]

    # figure
    plt.figure(figsize=(7,7))
    ax = plt.gca(); ax.set_aspect("equal"); ax.grid(True, linestyle=":")

    # LiDAR (ego@t0)
    if with_lidar and dataroot is not None:
        lidar_path = data["lidar_path"].item() if data["lidar_path"].shape == () else str(data["lidar_path"].tolist())
        pts = load_lidar_points_ego(lidar_path, sample_token, dataroot, version)
        if isinstance(pts, np.ndarray) and pts.shape[1] >= 2:
            ax.scatter(pts[:,0], pts[:,1], s=0.2, alpha=0.5, label="LiDAR t0 (ego@t0)")

    # map layers — match demo_nuscenes_v4.py styles
    added_lc = added_ld = added_rd = added_pc = added_sl = added_tl = False
    for xy in _iter_polylines(map_lane_center):
        ax.plot(xy[:,0], xy[:,1], linewidth=0.9, alpha=0.7, color=(0.2,0.5,0.9),
                label=None if added_lc else "lane_center", zorder=2.1)
        added_lc = True
    for xy in _iter_polylines(map_lane_div):
        ax.plot(xy[:,0], xy[:,1], linewidth=0.8, alpha=0.6, color=(0.5,0.5,0.5),
                label=None if added_ld else "lane_divider", zorder=2.0)
        added_ld = True
    for xy in _iter_polylines(map_road_div):
        ax.plot(xy[:,0], xy[:,1], linewidth=1.4, alpha=0.75, color=(0.25,0.25,0.25),
                label=None if added_rd else "road_divider", zorder=2.0)
        added_rd = True
    for poly in _iter_polylines(map_ped_xing):
        if poly.shape[0] >= 3:
            ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=0.15,
                            edgecolor=(0.0,0.6,0.0), facecolor=(0.0,0.8,0.0),
                            label=None if added_pc else "ped_crossing", zorder=1.8))
            added_pc = True
    for xy in _iter_polylines(map_stop_line):
        ax.plot(xy[:,0], xy[:,1], linewidth=1.8, alpha=0.85, color=(0.8,0.1,0.1),
                label=None if added_sl else "stop_line", zorder=2.2)
        added_sl = True
    tl = _iter_points(map_tl_points)
    if isinstance(tl, np.ndarray) and tl.size:
        ax.scatter(tl[:,0], tl[:,1], marker='*', s=36, alpha=0.9, color=(0.95,0.7,0.0),
                   label=None if added_tl else "traffic_light", zorder=2.3)
        added_tl = True

    # agents — dashed, t0 circle & arrow, future end arrow
    any_agent = False
    for i in range(Na):
        aid = str(A_id[i]) if i < len(A_id) else f"agent{i}"
        typ = str(A_type[i]) if i < len(A_type) else "agent.unknown"
        c   = _color_for_key(aid)
        xy_p, yaw_p, m_p = A_xy_h[i], A_yaw_h[i], A_m_h[i]
        xy_f, yaw_f, m_f = A_xy_f[i], A_yaw_f[i], A_m_f[i]
        xy_p = _boolean_mask_rows(xy_p, m_p)
        yaw_p = yaw_p[m_p] if yaw_p.shape[0] == m_p.shape[0] else yaw_p
        xy_f = _boolean_mask_rows(xy_f, m_f)
        yaw_f = yaw_f[m_f] if yaw_f.shape[0] == m_f.shape[0] else yaw_f
        if xy_p.size == 0 and xy_f.size == 0: continue
        segs = []
        if xy_p.size: segs.append(xy_p)
        if xy_f.size: segs.append(xy_f)
        path = np.vstack(segs) if segs else None
        if path is None or not visible(path, r):
            continue
        short = aid[:8]
        ax.plot(path[:,0], path[:,1], '--', linewidth=1.8, alpha=0.95, color=c,
                label=f"{typ.split('.')[-1]}[{short}]")
        if xy_p.size:
            x_now, y_now = xy_p[-1,0], xy_p[-1,1]
            ax.plot(x_now, y_now, 'o', markersize=4.5, color=c,
                    markeredgecolor='k', markeredgewidth=0.6)
            if yaw_p.size:
                th = float(yaw_p[-1])
                draw_heading(ax, x_now, y_now, th,  shaft_len=3.0, head_len=1.0, head_w=1.2, color=c, alpha=0.7, z=4)
        if xy_f.size and yaw_f.size:
            x_end, y_end = xy_f[-1,0], xy_f[-1,1]
            thf = float(yaw_f[-1])
            draw_heading(ax, x_end, y_end, thf, shaft_len=3.0, head_len=1.0, head_w=1.2, color=c, alpha=0.7, z=4)
        any_agent = True

    # ego — dashed, t0 and future arrows
    ego_color = _color_for_key("ego")
    segs = []
    if ego_xy_h_v.size: segs.append(ego_xy_h_v)
    if ego_xy_f_v.size: segs.append(ego_xy_f_v)
    ego_path = np.vstack(segs) if segs else None
    if ego_path is not None:
        ax.plot(ego_path[:,0], ego_path[:,1], '--', linewidth=2.2, color=ego_color, label="ego")
    print(f"ego_xy_h_v.size = {ego_xy_h_v.size}")
    if ego_xy_h_v.size:
        x0, y0 = ego_xy_h_v[-1,0], ego_xy_h_v[-1,1]
        ax.plot(x0, y0, 'o', markersize=5.0, color=ego_color, markeredgecolor='k', markeredgewidth=0.8)
        if ego_yaw_h_v.size:
            th = float(ego_yaw_h_v[-1])
            draw_heading(ax, x0, y0, th,
                        shaft_len=5.0, head_len=1.0, head_w=1.5,
                        color=ego_color, alpha=0.8, z=5)

    print(f"ego_xy_f_v.size = {ego_xy_f_v.size}")
    if ego_xy_f_v.size and ego_yaw_f_v.size:
        xf, yf, thf = ego_xy_f_v[-1,0], ego_xy_f_v[-1,1], float(ego_yaw_f_v[-1])
        draw_heading(ax, xf, yf, thf,
                 shaft_len=5.0, head_len=1.0, head_w=1.5,
                 color=ego_color, alpha=0.8, z=5)        

    # ego@t0 axes & title
    ax.plot([0],[0], marker="o", markersize=4, color="k")
    ax.arrow(0,0,10,0, length_includes_head=True, head_width=3, head_length=4, alpha=0.6)  # x forward
    ax.arrow(0,0,0,10, length_includes_head=True, head_width=3, head_length=4, alpha=0.6)  # y left
    ax.text(12, 0, "x_fwd", fontsize=8); ax.text(0, 12, "y_left", fontsize=8)

    # axes/legend same as demo
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_xlabel("X (m) — ego@t0"); ax.set_ylabel("Y (m) — ego@t0")
    ax.set_title("BEV Preview (all in ego@t0)")
    if any_agent or ego_xy_h_v.size or ego_xy_f_v.size:
        #ax.legend(loc="upper right", fontsize=8, ncol=1)
        leg = ax.legend(
            loc="upper right",
            fontsize=6,          # 字更小
            ncol=2,              # 需要可改成 2，垂直高度會更短
            markerscale=0.8,     # 縮小點/星號
            handlelength=1.2,    # 線段長度短一點
            handletextpad=0.4,   # 線與文字間距
            borderpad=0.25,      # 框與內容內縮
            labelspacing=0.25,   # 行距
            columnspacing=0.8,   # 欄與欄的距離（ncol>1 才用得到）
            framealpha=0.95
        )
        leg.get_frame().set_linewidth(0.6)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int)
    g.add_argument("--sample_token", type=str)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--dataroot", required=True, help="nuScenes dataroot (for lidar extrinsics)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--no_lidar", action="store_true", help="disable LiDAR overlay")
    args = ap.parse_args()

    row = load_manifest_row(args.manifest, idx=args.idx, sample_token=args.sample_token)
    npz_path = row["npz"]

    npz = np.load(npz_path, allow_pickle=True)
    print("=== Manifest row ==="); print(json.dumps(row, indent=2))
    print("\n=== NPZ summary ==="); print(summarize_npz(npz))

    print(f"\nRendering to {args.out_png} …")
    plot_from_npz(
        npz_path,
        args.out_png,
        dataroot=args.dataroot,
        version=args.version,
        with_lidar=(not args.no_lidar)
    )
    print("Done.")

if __name__ == "__main__":
    main()
