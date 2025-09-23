"""
Render a quick BEV preview with everything transformed into the ego@t0 frame.
- LiDAR t0 points: lidar -> ego(t0) -> world -> ego@t0
- Ego history (world xy,yaw): world -> ego@t0 ; yaw 相對 t0
- Agents history (world xy,yaw,mask): 先用 mask 過濾，再 world -> ego@t0；yaw 相對 t0

Coordinate convention of ego@t0: x forward, y left, z up.
"""

import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def visible(xy, r): 
    return np.any((np.abs(xy[:,0])<=r) & (np.abs(xy[:,1])<=r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes dataroot (contains samples/, sweeps/, maps/, v1.0-*)")
    ap.add_argument("--sample_token", required=True, help="Keyframe sample token")
    ap.add_argument("--out_png", default="nuscenes_preview.png")
    ap.add_argument("--past_sec", type=float, default=4.0, help="history window in seconds")
    ap.add_argument("--map_radius_m", type=float, default=80.0, help="BEV radius around ego for plotting")
    ap.add_argument("--with_lidar", action="store_true", help="Draw t0 LIDAR_TOP points")
    args = ap.parse_args()

    # Lazy imports
    from pyquaternion import Quaternion
    from nuscenes import NuScenes
    from trmp.datasets.nuscenes import load_sample

    # 1) Load unified sample (images paths, lidar path, histories in WORLD XY, etc.)
    sample = load_sample(args.dataroot, args.sample_token, past_sec=args.past_sec, map_radius_m=args.map_radius_m)
    t0 = sample["t0"]

    # world <- ego(t0)  /  ego <- lidar
    #R_we = Quaternion(t0["ego_rot"]).rotation_matrix
    #t_we = np.array(t0["ego_trans"], dtype=np.float32)
    R_we = t0["ego_rot"]
    t_we = t0["ego_trans"]
    R_el = t0["calib_rot"]
    t_el = t0["calib_trans"]
    lidar_path = t0["lidar_filename"]

    # 2) Pull transforms at t0 (from the LIDAR_TOP keyframe)
    #nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    #s = nusc.get("sample", args.sample_token)
    #sd_lidar = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    #cal = nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])   # ego <- lidar (extrinsic)
    #pose_t0 = nusc.get("ego_pose", sd_lidar["ego_pose_token"])                 # world <- ego(t0)

    # world <- ego(t0)
    #R_we = Quaternion(pose_t0["rotation"]).rotation_matrix  # [3,3]
    #t_we = np.array(pose_t0["translation"], dtype=np.float32)  # [3]

    # Helpers
    def world_to_ego_t0_xy(xy_world: np.ndarray) -> np.ndarray:
        """[T,2] world XY -> ego@t0 XY"""
        if xy_world.size == 0:
            return xy_world
        d = xy_world - t_we[:2]                    # subtract world origin (ego@t0 in world)
        R2 = R_we[:2, :2]                          # world<-ego(t0) rotation (3x3) -> take XY block
        # ego@t0 <- world is R_we^T on vectors, so here use d @ R2 (right-mul by R^T is left-mul by R)
        return (d @ R2)

    # 3) LiDAR points at t0 -> transform to ego@t0 (so it aligns with histories)
    pts_ego = np.zeros((0, 4), dtype=np.float32)
    if args.with_lidar:
        lid = sample.get("lidar", {})
        if isinstance(lid, dict) and "path" in lid and os.path.exists(lid["path"]):
            pc = LidarPointCloud.from_file(lid["path"])  # [4, N]
            pts_lidar = pc.points[:3, :].T               # [N,3] in LIDAR frame
            # lidar -> ego(t0)
            pts_ego_t0 = (pts_lidar @ R_el.T) + t_el     # [N,3] in ego(t0)
            # ego(t0) -> world
            pts_world = (pts_ego_t0 @ R_we.T) + t_we     # [N,3] in world
            # world -> ego@t0  (bring everything to common frame)
            pts_ego_xy = world_to_ego_t0_xy(pts_world[:, :2])
            pts_ego = np.c_[pts_ego_xy, pts_world[:, 2], np.zeros((pts_world.shape[0], 1), dtype=np.float32)]
            # ^ 保存 z 與一個假的 intensity 欄位，成 [N,4]

    # 4) Transform histories from WORLD to ego@t0 (use mask for agents; convert yaw to "relative to t0")
    # ==== 把 ego 的 yaw 從 world 轉成相對於 t0 的 yaw（ego@t0） ====
    # Ego: [T,3] (world) -> ego@t0 ; yaw_rel
    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    eh = sample.get("ego_history")
    if isinstance(eh, np.ndarray) and eh.size and eh.shape[1] >= 3:
        # yaw relative to t0 in world
        yaw_world = eh[:, 2].copy()
        yaw0_world = yaw_world[-1]          # t0 的世界朝向
        yaw_rel = wrap_pi(yaw_world - yaw0_world)
        sample["ego_history"][:, :2] = world_to_ego_t0_xy(eh[:, :2])
        sample["ego_history"][:, 2] = yaw_rel

    # Agents: dict[instance] -> {xy:[T,2], yaw:[T], mask:[T], ...}
    r = float(args.map_radius_m)
    KEEP_PREFIX = ("vehicle.", "human.pedestrian")  # 可改成 None 表示全部顯示
    for rec in sample.get("agents_history", {}).values():
        typ = rec.get("type", "")
        if KEEP_PREFIX and (not typ.startswith(KEEP_PREFIX)):
            # 跳過 movable_object.*, traffic_cone 等
            rec["xy"] = np.zeros((0, 2), dtype=np.float32)
            rec["yaw"] = np.zeros((0,), dtype=np.float32)
            rec["mask"] = np.zeros((0,), dtype=bool)
            continue

        xy_w = rec.get("xy")         # [T,2] world
        m = rec.get("mask")          # [T] bool
        yaw_w = rec.get("yaw")       # [T] world heading

        if not (isinstance(xy_w, np.ndarray) and xy_w.size):
            rec["xy"] = np.zeros((0, 2), dtype=np.float32)
            rec["yaw"] = np.zeros((0,), dtype=np.float32)
            rec["mask"] = np.zeros((0,), dtype=bool)
            continue

        # 只保留有效時間步
        if isinstance(m, np.ndarray) and m.dtype == bool and m.shape[0] == xy_w.shape[0]:
            xy_w = xy_w[m]
            yaw_w = yaw_w[m] if (isinstance(yaw_w, np.ndarray) and yaw_w.shape[0] == m.shape[0]) else None
        # 轉到 ego@t0
        xy_e = world_to_ego_t0_xy(xy_w)
        rec["xy"] = xy_e
        # agent 相對 t0 的 yaw（和 ego 做法一致）
        if isinstance(yaw_w, np.ndarray) and yaw_w.size:
            rec["yaw"] = wrap_pi(yaw_w - yaw0_world)
        else:
            rec["yaw"] = np.zeros((xy_e.shape[0],), dtype=np.float32)
        # 這裡不再保留 mask（已經篩掉無效步）；若想保留，可設 rec["mask"] = np.ones(len(xy_e), bool)

    '''
    # Agents history: dict of {instance_token: {"xy": [T,2], ...}}
    for rec in sample.get("agents_history", {}).values():
        xy = rec.get("xy")
        if isinstance(xy, np.ndarray) and xy.size:
          rec["xy"] = world_to_ego_t0_xy(xy)
    '''

    # 5) Draw BEV
    plt.figure(figsize=(7,7))
    ax = plt.gca(); ax.set_aspect("equal"); ax.grid(True, linestyle=":")

    # LiDAR (ego@t0)
    if pts_ego.size and pts_ego.shape[1] >= 2:
        ax.scatter(pts_ego[:, 0], pts_ego[:, 1], s=0.2, alpha=0.5, label="LiDAR t0 (ego@t0)")

    # Agents history（每個 agent 都有 legend；只畫落在視野內的）
    any_agent = False
    for aid, rec in sample.get("agents_history", {}).items():
        xy = rec.get("xy")
        if not (isinstance(xy, np.ndarray) and xy.size and visible(xy, r)):
            continue
        typ = rec.get("type", "").split(".")[-1] or "agent"
        short = str(aid)[:8]
        ax.plot(xy[:, 0], xy[:, 1], '--', linewidth=1.2, alpha=0.95, label=f"{typ}[{short}]")
        # 在最後一點畫朝向箭頭（3 m）
        yaw_rel = rec.get("yaw")
        if isinstance(yaw_rel, np.ndarray) and yaw_rel.size:
            x1, y1, th = xy[-1, 0], xy[-1, 1], yaw_rel[-1]
            ax.arrow(x1, y1, 3*np.cos(th), 3*np.sin(th),
                     length_includes_head=True, head_width=1.2, head_length=2.0, alpha=0.7)
        any_agent = True

    '''
    # Agents history (ego@t0) — 每個 agent 都有自己的 legend
    any_agent = False
    for aid, rec in sample.get("agents_history", {}).items():
        typ = rec.get("type", "")
        if not typ.startswith(KEEP_PREFIX):
            continue   # 跳過 movable_object.* 等
        xy = rec.get("xy")
        if isinstance(xy, np.ndarray) and xy.size and visible(xy, r):
            short = str(aid)[:8]
            ax.plot(xy[:,0], xy[:,1], '--', linewidth=1.0, alpha=0.9, label=f"{typ.split('.')[-1]}[{short}]")
            any_agent = True
    '''

    # Ego history (ego@t0)
    if isinstance(eh, np.ndarray) and eh.size:
        ax.plot(eh[:, 0], eh[:, 1], "--", linewidth=2.0, label="ego (ego@t0)")
        # 在 t0 畫一個短箭頭（長度 5m）
        x0, y0, yaw0_rel = eh[-1, 0], eh[-1, 1], eh[-1, 2]
        ax.arrow(x0, y0, 5*np.cos(yaw0_rel), 5*np.sin(yaw0_rel),
                length_includes_head=True, head_width=1.5, head_length=2.5,
                fc="magenta", ec="magenta", alpha=0.8)

    # Origin marker & heading (ego@t0 origin)
    ax.plot([0], [0], marker="o", markersize=4, color="k")
    ax.arrow(0, 0, 10, 0, length_includes_head=True, head_width=3, head_length=4, alpha=0.6)  # x-forward
    ax.arrow(0, 0, 0, 10, length_includes_head=True, head_width=3, head_length=4, alpha=0.6)  # y-left
    ax.text(12, 0, "x_fwd", fontsize=8); ax.text(0, 12, "y_left", fontsize=8)

    # Axis & title
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_xlabel("X (m) — ego@t0"); ax.set_ylabel("Y (m) — ego@t0"); ax.set_title("BEV Preview (all in ego@t0)")
    if any_agent or pts_ego.size > 0 or (isinstance(eh, np.ndarray) and eh.size):
        ax.legend(loc="upper right", fontsize=8, ncol=1)    

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved preview to {args.out_png}")

if __name__ == "__main__":
    main()