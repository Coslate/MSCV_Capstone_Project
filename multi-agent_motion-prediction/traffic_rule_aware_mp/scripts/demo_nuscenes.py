import argparse, yaml
from trmp.datamodule import DataConfig, DataModule
from trmp.visualize.bev_preview import plot_bev

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/nuscenes.yaml")
    ap.add_argument("--dataroot", type=str, required=False, help="override nuScenes dataroot")
    ap.add_argument("--sample_token", type=str, required=False, help="override sample_token")
    ap.add_argument("--out_png", type=str, default="nuscenes_preview.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.dataroot: cfg["root"] = args.dataroot
    if args.sample_token: cfg["sample_token"] = args.sample_token

    dm = DataModule(DataConfig(**cfg))
    sample = next(dm.iter_samples())
    # For preview, we won't actually load images/pointcloud files; just plot agents if present.
    plot_bev(sample, out_png=args.out_png)
    print(f"Saved preview to {args.out_png}")
