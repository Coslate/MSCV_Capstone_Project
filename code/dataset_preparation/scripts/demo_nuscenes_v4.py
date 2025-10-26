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
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuscenes.map_expansion.map_api import NuScenesMap
from matplotlib.patches import Polygon as MplPolygon

from trmp.datasets.nuscenes import load_sample
from trmp.utils.map_helper import (
    _lane_or_connector_centerline,
    _line_xy_from_token,
    _polygon_xy_from_token,
    _stop_line_polyline,
    _traffic_light_point,
    color_for_key,
    visible,
    report_stop_line_quality,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes dataroot (contains samples/, sweeps/, maps/, v1.0-*)")
    ap.add_argument("--sample_token", required=True, help="Keyframe sample token")
    ap.add_argument("--out_png", default="nuscenes_preview.png")
    ap.add_argument("--past_sec", type=float, default=4.0, help="history window in seconds")
    ap.add_argument("--map_radius_m", type=float, default=80.0, help="BEV radius around ego for plotting")
    ap.add_argument("--with_lidar", action="store_true", help="Draw t0 LIDAR_TOP points")
    ap.add_argument("--future_sec", type=float, default=6.0, help="future window in seconds")

    args = ap.parse_args()

    # 1) Load unified sample (images paths, lidar path, histories in WORLD XY, etc.)
    sample = load_sample(args.dataroot, args.sample_token, past_sec=args.past_sec, future_sec=args.future_sec, map_radius_m=args.map_radius_m)
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
        """[N,2 or 3] WORLD -> ego@t0 (row-vector). 只用 XY 做平面轉換。"""
        if not isinstance(xy_world, np.ndarray) or xy_world.size == 0:
            return xy_world
        xy = np.asarray(xy_world, dtype=np.float32)
        xy = np.atleast_2d(xy)
        xy = xy[:, :2]  # 關鍵：丟掉多出來的維度（例如 yaw）
        R2 = np.asarray(R_we, dtype=np.float32)[:2, :2]  # world_from_ego(t0)
        tw = np.asarray(t_we, dtype=np.float32)[:2]
        return (xy - tw) @ R2

    '''
    def world_to_ego_t0_xy(xy_world: np.ndarray) -> np.ndarray:
        """[N,2] WORLD XY -> ego@t0 XY (row-vector form).  x_e = (x_w - t_we) @ R_we[:2,:2]"""
        if xy_world.size == 0:
            return xy_world
        d = xy_world - t_we[:2]                    # subtract world origin (ego@t0 in world)
        R2 = R_we[:2, :2]                          # world<-ego(t0) rotation (3x3) -> take XY block
        # ego@t0 <- world is R_we^T on vectors, so here use d @ R2 (right-mul by R^T is left-mul by R)
        return (d @ R2)
    '''

    # -- Build NuScenesMap and extract layers in WORLD, then map to ego@t0 --
    nmap = NuScenesMap(dataroot=args.dataroot, map_name=sample["city_or_map_id"])


    # --- quick statistics: how many lanes actually carry usable boundaries? ---
    has_seg_nodes = has_segments = has_centerline_token = has_inline_center = 0
    lane_like = sample["map"].get("lane_center", [])
    for tok in lane_like:
        rec = None
        for tb in ("lane", "lane_connector"):
            try:
                rec = nmap.get(tb, tok)
                break
            except KeyError:
                continue
        if rec is None:
            continue

        ln = rec.get('left_lane_divider_segment_nodes') or []
        rn = rec.get('right_lane_divider_segment_nodes') or []
        ls = rec.get('left_lane_divider_segments') or []
        rs = rec.get('right_lane_divider_segments') or []
        if ln and rn:
            has_seg_nodes += 1
        if ls and rs:
            has_segments += 1

        if rec.get('centerline_line_token') or rec.get('line_token'):
            has_centerline_token += 1
        if rec.get('centerline') or rec.get('baseline_path'):
            has_inline_center += 1

    print(
        f"[lane stats] total={len(lane_like)}  "
        f"seg_nodes L&R={has_seg_nodes}  segments L&R={has_segments}  "
        f"centerline_token={has_centerline_token}  inline_center={has_inline_center}"
    )

    # 2.1 lane center = lane + lane_connector
    lane_center_polylines = []
    lane_tokens = []
    lane_tokens += sample["map"].get("lane_center", [])
    lane_tokens += sample["map"].get("lane_connector", [])

    for tok in lane_tokens:
        xy_w = _lane_or_connector_centerline(nmap, tok, ds=1.0)
        if xy_w is None or xy_w.shape[0] < 2:
            continue
        lane_center_polylines.append(world_to_ego_t0_xy(xy_w))

    # 2.2 lane_divider
    lane_divider_polylines = []
    for tok in sample["map"].get("lane_divider", []):
        rec = nmap.get('lane_divider', tok)
        xy_w = _line_xy_from_token(nmap, rec.get('line_token', None))
        if xy_w is not None and xy_w.shape[0] >= 2:
            lane_divider_polylines.append(world_to_ego_t0_xy(xy_w))

    # 2.3 road_divider
    road_divider_polylines = []
    for tok in sample["map"].get("road_divider", []):
        rec = nmap.get('road_divider', tok)
        xy_w = _line_xy_from_token(nmap, rec.get('line_token', None))
        if xy_w is not None and xy_w.shape[0] >= 2:
            road_divider_polylines.append(world_to_ego_t0_xy(xy_w))

    # 2.4 ped_crossing（polygon）
    ped_crossing_polys = []
    for tok in sample["map"].get("ped_crossing", []):
        rec = nmap.get('ped_crossing', tok)
        xy_w = _polygon_xy_from_token(nmap, rec.get('polygon_token', None))
        if xy_w is not None and xy_w.shape[0] >= 3:
            ped_crossing_polys.append(world_to_ego_t0_xy(xy_w))

    # 2.5 stop_line（多樣 fallback）
    stop_line_polylines = []
    for tok in sample["map"].get("stop_line", []):
        xy_w = _stop_line_polyline(nmap, tok)
        if xy_w is not None and xy_w.shape[0] >= 2:
            stop_line_polylines.append(world_to_ego_t0_xy(xy_w))

    # 2.6 traffic_light（point）
    traffic_light_points = []
    for tok in sample["map"].get("traffic_light", []):
        xy_w = _traffic_light_point(nmap, tok)
        if xy_w is not None:
            traffic_light_points.append(world_to_ego_t0_xy(xy_w))

    # 3) LiDAR points at t0 -> transform to ego@t0 (so it aligns with histories)
    pts_ego = np.zeros((0, 4), dtype=np.float32)
    if args.with_lidar:
        lid = sample.get("lidar", {})
        if isinstance(lid, dict) and "path" in lid and os.path.exists(lid["path"]):
            pc = LidarPointCloud.from_file(lid["path"])        # [4, N]
            pts_lidar = pc.points[:3, :].T                     # [N,3] in LIDAR frame
            intensity = pc.points[3, :].astype(np.float32)     # [N] reflectance
            # lidar -> ego(t0)  (ego(t0) == ego@t0)
            pts_ego_t0 = (pts_lidar @ R_el.T) + t_el           # [N,3]
            # 直接用 ego@t0 的 (x,y,z)，不要再經過 world
            pts_ego = np.c_[pts_ego_t0, intensity[:, None]]    # [N,4] = (x,y,z,intensity)

    # 4) Transform histories from WORLD to ego@t0 (use mask for agents; convert yaw to "relative to t0")
    # ==== 把 ego 的 yaw 從 world 轉成相對於 t0 的 yaw（ego@t0） ====
    # Ego: [T,3] (world) -> ego@t0 ; yaw_rel
    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    eh = sample.get("ego_history")
    if isinstance(eh, np.ndarray) and eh.size and eh.shape[1] >= 3:
        # yaw relative to t0 in world
        yaw0_world = float(eh[-1, 2])                    # t0 的世界朝向
        xy_e   = world_to_ego_t0_xy(eh[:, :2].copy())
        yaw_rel = wrap_pi(eh[:, 2] - yaw0_world)
        eh[:, :2] = xy_e
        eh[:, 2]  = yaw_rel

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

    # ==== FUTURE：把 ego & agents 的未來從 world 轉到 ego@t0，yaw 轉相對 t0 ====
    # Ego future
    ef = sample.get("ego_future")
    ego_future_xy_e = np.zeros((0,2), dtype=np.float32)
    ego_future_yaw_rel = np.zeros((0,), dtype=np.float32)
    if isinstance(ef, np.ndarray) and ef.size and ef.shape[1] >= 3:
        ego_future_xy_e = world_to_ego_t0_xy(ef[:, :2])
        ego_future_yaw_rel = wrap_pi(ef[:, 2] - yaw0_world)

    # Agents future（與 history 使用相同 instance 集合、相同顏色）
    agents_future_e = {}
    for aid, rec in sample.get("agents_future", {}).items():
        # 同樣用 KEEP_PREFIX 篩類別（若要一致，建議沿用你上面 history 的篩法）
        typ = rec.get("type", "")
        if KEEP_PREFIX and (not typ.startswith(KEEP_PREFIX)):
            continue

        xy_w_f = rec.get("xy")      # [Tf,2] world
        m_f    = rec.get("mask")    # [Tf]    bool
        yaw_wf = rec.get("yaw")     # [Tf]

        if not (isinstance(xy_w_f, np.ndarray) and xy_w_f.size):
            continue

        if isinstance(m_f, np.ndarray) and m_f.dtype == bool and m_f.shape[0] == xy_w_f.shape[0]:
            xy_w_f = xy_w_f[m_f]
            yaw_wf = yaw_wf[m_f] if (isinstance(yaw_wf, np.ndarray) and yaw_wf.shape[0] == m_f.shape[0]) else None

        xy_e_f = world_to_ego_t0_xy(xy_w_f)
        yaw_rel_f = wrap_pi(yaw_wf - yaw0_world) if isinstance(yaw_wf, np.ndarray) and yaw_wf.size else np.zeros((xy_e_f.shape[0],), dtype=np.float32)
        agents_future_e[aid] = {"xy": xy_e_f, "yaw": yaw_rel_f, "type": typ}        

    print("[map tokens]",
        "lane:", len(sample["map"].get("lane_center", [])),
        "lane_conn:", len(sample["map"].get("lane_connector", [])),
        "ld:", len(sample["map"].get("lane_divider", [])),
        "rd:", len(sample["map"].get("road_divider", [])),
        "pc:", len(sample["map"].get("ped_crossing", [])),
        "sl:", len(sample["map"].get("stop_line", [])),
        "tl:", len(sample["map"].get("traffic_light", [])))
    print("[draw counts]",
        "lc:", len(lane_center_polylines),
        "ld:", len(lane_divider_polylines),
        "rd:", len(road_divider_polylines),
        "pc:", len(ped_crossing_polys),
        "sl:", len(stop_line_polylines),
        "tl:", sum(p.shape[0] for p in traffic_light_points))

    # 5) Draw BEV
    plt.figure(figsize=(7,7))
    ax = plt.gca(); ax.set_aspect("equal"); ax.grid(True, linestyle=":")

    # LiDAR (ego@t0)
    if pts_ego.size and pts_ego.shape[1] >= 2:
        ax.scatter(pts_ego[:, 0], pts_ego[:, 1], s=0.2, alpha=0.5, label="LiDAR t0 (ego@t0)")

    # HD map（在線雲之上）
    added_lc = added_ld = added_rd = added_pc = added_sl = added_tl = False
    for xy in lane_center_polylines:
        ax.plot(xy[:,0], xy[:,1], linewidth=0.9, alpha=0.7, color=(0.2,0.5,0.9),
                label=None if added_lc else "lane_center", zorder=2.1)
        added_lc = True

    for xy in lane_divider_polylines:
        ax.plot(xy[:,0], xy[:,1], linewidth=0.8, alpha=0.6, color=(0.5,0.5,0.5),
                label=None if added_ld else "lane_divider", zorder=2.0)
        added_ld = True

    for xy in road_divider_polylines:
        ax.plot(xy[:,0], xy[:,1], linewidth=1.4, alpha=0.75, color=(0.25,0.25,0.25),
                label=None if added_rd else "road_divider", zorder=2.0)
        added_rd = True

    for poly in ped_crossing_polys:
        if poly.shape[0] < 3:  # 避免病態多邊形
            continue
        ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=0.15,
                            edgecolor=(0.0,0.6,0.0), facecolor=(0.0,0.8,0.0),
                            label=None if added_pc else "ped_crossing", zorder=1.8))
        added_pc = True

    for xy in stop_line_polylines:
        ax.plot(xy[:,0], xy[:,1], linewidth=1.8, alpha=0.85, color=(0.8,0.1,0.1),
                label=None if added_sl else "stop_line", zorder=2.2)
        added_sl = True

    if traffic_light_points:  # list of [M_i, 2] arrays
        pts = np.vstack(traffic_light_points)
        ax.scatter(pts[:,0], pts[:,1], marker='*', s=36, alpha=0.9, color=(0.95,0.7,0.0),
                label=None if added_tl else "traffic_light", zorder=2.3)
        added_tl = True        

    # Agents — past+future 用同一種線型與同一個 legend 條目；t0 畫圓圈；t0 與未來末端畫 heading
    any_agent = False
    for aid, rec in sample.get("agents_history", {}).items():
        typ = rec.get("type", "")
        if KEEP_PREFIX and (not typ.startswith(KEEP_PREFIX)):
            continue

        c = color_for_key(aid)
        short = str(aid)[:8]

        # PAST（已在上面用 mask 篩過，且已轉 ego@t0）
        xy_p  = rec.get("xy")
        yaw_p = rec.get("yaw")

        # FUTURE（已在上面轉 ego@t0）
        f = agents_future_e.get(aid, None)
        xy_f  = f.get("xy")  if f is not None else None
        yaw_f = f.get("yaw") if f is not None else None

        # 若 past+future 都沒有點就跳過
        has_p = isinstance(xy_p, np.ndarray) and xy_p.size
        has_f = isinstance(xy_f, np.ndarray) and xy_f.size
        if not (has_p or has_f):
            continue

        # 串成一條連續 polyline（past: oldest->t0，future: t1->end）
        segs = []
        if has_p: segs.append(xy_p)
        if has_f: segs.append(xy_f)
        path = np.vstack(segs) if segs else None
        if path is None or not visible(path, r):
            continue

        # 單一 legend 條目（不再分 past/future）
        ax.plot(path[:, 0], path[:, 1], '--', linewidth=1.8, alpha=0.95, color=c,
                label=f"{typ.split('.')[-1]}[{short}]")

        # t0 圓圈（取 past 的最後一點）
        if has_p:
            x_now, y_now = xy_p[-1, 0], xy_p[-1, 1]
            ax.plot(x_now, y_now, 'o', markersize=4.5, color=c,
                    markeredgecolor='k', markeredgewidth=0.6)
            # t0 heading
            if isinstance(yaw_p, np.ndarray) and yaw_p.size:
                th = yaw_p[-1]
                ax.arrow(x_now, y_now, 3*np.cos(th), 3*np.sin(th),
                         length_includes_head=True, head_width=1.2, head_length=2.0,
                         alpha=0.7, color=c)

        # 未來末端 heading（若有未來）
        if has_f:
            x_end, y_end = xy_f[-1, 0], xy_f[-1, 1]
            if isinstance(yaw_f, np.ndarray) and yaw_f.size:
                thf = yaw_f[-1]
                ax.arrow(x_end, y_end, 3*np.cos(thf), 3*np.sin(thf),
                         length_includes_head=True, head_width=1.2, head_length=2.0,
                         alpha=0.7, color=c)

        any_agent = True

    # Ego Part
    ego_color = color_for_key('ego')

    # 合併 ego 的 past+future
    ego_segs = []
    if isinstance(eh, np.ndarray) and eh.size:
        ego_segs.append(eh[:, :2])  # past 包含 t0
    if ego_future_xy_e.shape[0] > 0:
        ego_segs.append(ego_future_xy_e)  # future
    ego_path = np.vstack(ego_segs) if ego_segs else None

    print(f"ego_path.shape = {ego_path.shape}")
    if ego_path is not None and ego_path.size:
        ax.plot(ego_path[:, 0], ego_path[:, 1], '--', linewidth=2.2, color=ego_color, label="ego")

    # t0 圓圈 & heading（相對 yaw 已在 eh 內）
    if isinstance(eh, np.ndarray) and eh.size:
        x0, y0, yaw0_rel = eh[-1, 0], eh[-1, 1], eh[-1, 2]
        ax.plot(x0, y0, 'o', markersize=5.0, color=ego_color, markeredgecolor='k', markeredgewidth=0.8)
        ax.arrow(x0, y0, 5*np.cos(yaw0_rel), 5*np.sin(yaw0_rel),
                 length_includes_head=True, head_width=1.5, head_length=2.5,
                 fc=ego_color, ec=ego_color, alpha=0.8)

    # 未來末端 heading（若有）
    if ego_future_xy_e.shape[0] > 0 and ego_future_yaw_rel.size:
        xf, yf = ego_future_xy_e[-1, 0], ego_future_xy_e[-1, 1]
        thf = ego_future_yaw_rel[-1]
        ax.arrow(xf, yf, 5*np.cos(thf), 5*np.sin(thf),
                 length_includes_head=True, head_width=1.5, head_length=2.5,
                 fc=ego_color, ec=ego_color, alpha=0.8)

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

#-------------------------------- Sanity Check -------------------------------------#
    report_stop_line_quality(
        stop_line_polylines,
        lane_center_polylines,
        r,                 # 你 main 裡的 map 半徑
        near_lane_r=35.0,  # 可調參數
        clip_L=4.0,
        length_clip_thresh=6.0,
        top_k=3
    )
#-----------------------------------------------------------------------------------#        

if __name__ == "__main__":
    main()