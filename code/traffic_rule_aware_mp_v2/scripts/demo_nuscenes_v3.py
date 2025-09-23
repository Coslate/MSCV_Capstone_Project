import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--sample_token", required=True)
    ap.add_argument("--out_png", default="nuscenes_preview.png")
    ap.add_argument("--past_sec", type=float, default=3.0, help="history window in seconds")
    ap.add_argument("--map_radius_m", type=float, required=True, default=80.0, help="radius around ego to include in map")
    ap.add_argument("--with_lidar", action="store_true", help="draw t0 LIDAR_TOP points")
    args = ap.parse_args()

    # import after parsing to avoid heavy init on --help
    from trmp.datasets.nuscenes import load_sample
    sample = load_sample(args.dataroot, args.sample_token, past_sec=args.past_sec, map_radius_m=args.map_radius_m)

    # Optionally read the t0 lidar file into Nx4 [x,y,z,intensity]
    if args.with_lidar:
        from nuscenes.utils.data_classes import LidarPointCloud
        lid = sample.get("lidar", {})
        if isinstance(lid, dict) and "path" in lid and os.path.exists(lid["path"]):
            pc = LidarPointCloud.from_file(lid["path"])  # [4, N]
            sample["lidar"] = pc.points.T[:, :4].astype(np.float32)
        else:
            sample["lidar"] = np.zeros((0,4), dtype=np.float32)
    else:
        sample["lidar"] = np.zeros((0,4), dtype=np.float32)

    # --- simple BEV plot (包含 agents 歷史 + ego 歷史 + (可選)點雲) ---
    plt.figure(figsize=(6,6))
    ax = plt.gca(); ax.set_aspect("equal"); ax.grid(True, linestyle=":")

    # lidar
    pts = sample.get("lidar")
    if isinstance(pts, np.ndarray) and pts.size and pts.shape[1] >= 2:
        ax.scatter(pts[:,0], pts[:,1], s=0.2, alpha=0.5, label="LiDAR t0")

    # agents history
    any_agent = False
    for aid, rec in sample.get("agents_history", {}).items():
        xy = rec.get("xy")
        if isinstance(xy, np.ndarray) and xy.size:
            ax.plot(xy[:,0], xy[:,1], linewidth=1.0, alpha=0.9)
            any_agent = True

    # ego history
    eh = sample.get("ego_history")
    if isinstance(eh, np.ndarray) and eh.size:
        ax.plot(eh[:,0], eh[:,1], "--", linewidth=2.0, label="ego")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("BEV Preview")
    if any_agent or (isinstance(eh, np.ndarray) and eh.size) or (isinstance(pts, np.ndarray) and pts.size):
        ax.legend(loc="upper right", fontsize=8)
    else:
        # 避免空白圖：給個 80m 視野
        ax.set_xlim(-80, 80); ax.set_ylim(-80, 80)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved preview to {args.out_png}")

if __name__ == "__main__":
    main()