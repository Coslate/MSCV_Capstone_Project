import argparse, yaml, numpy as np
from trmp.datamodule import DataConfig, DataModule
from trmp.visualize.bev_preview import plot_bev

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--sample_token", type=str, required=True)
    ap.add_argument("--past_seconds", type=float, required=True, default=3.0)
    ap.add_argument("--map_radius_m", type=float, required=True, default=80.0)
    ap.add_argument("--out_png", type=str, default="nuscenes_preview.png")
    args = ap.parse_args()

    dm = DataModule(DataConfig(name="nuscenes", root=args.dataroot, sample_token=args.sample_token, past_seconds=args.past_seconds, map_readius_m=args.map_radius_m))
    sample = next(dm.iter_samples())
    # 視覺化示意（此版不讀取點雲檔；只畫 agent 歷史）
    sample["lidar"] = np.zeros((0,4), dtype=np.float32)
    plot_bev(sample, out_png=args.out_png)
    print(f"Saved preview to {args.out_png}")
