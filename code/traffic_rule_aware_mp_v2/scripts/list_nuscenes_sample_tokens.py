import argparse
from nuscenes import NuScenes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot)
    for i, s in enumerate(nusc.sample):
        print(s['token'])
        if i+1 >= args.n: break
