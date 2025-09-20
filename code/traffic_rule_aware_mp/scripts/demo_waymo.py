import argparse, yaml
from trmp.datamodule import DataConfig, DataModule

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/waymo_perception.yaml")
    ap.add_argument("--tfrecord", type=str, required=False)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.tfrecord: cfg["waymo_tfrecord"] = args.tfrecord

    dm = DataModule(DataConfig(**cfg))
    sample = next(dm.iter_samples())
    print("Loaded one Waymo frame (skeleton): keys=", list(sample.keys()))
