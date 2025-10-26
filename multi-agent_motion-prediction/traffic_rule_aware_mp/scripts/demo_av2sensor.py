import argparse, yaml
from trmp.datamodule import DataConfig, DataModule
from trmp.visualize.bev_preview import plot_bev
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/av2sensor.yaml")
    ap.add_argument("--log_dir", type=str, required=False, help="override AV2 log dir")
    ap.add_argument("--out_png", type=str, default="av2_preview.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.log_dir: cfg["av2_log_dir"] = args.log_dir

    dm = DataModule(DataConfig(**cfg))
    sample = next(dm.iter_samples())
    # Ensure lidar is ndarray for plotting in this skeleton
    if isinstance(sample["lidar"], dict):
        sample["lidar"] = np.zeros((0,4))
    plot_bev(sample, out_png=args.out_png)
    print(f"Saved preview to {args.out_png}")
