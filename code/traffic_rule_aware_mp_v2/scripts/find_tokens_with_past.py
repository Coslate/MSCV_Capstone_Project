import argparse

def main():
    ap = argparse.ArgumentParser(
        description="List nuScenes sample_tokens that have at least S seconds of past keyframe history."
    )
    ap.add_argument("--dataroot", required=True, help="Path to nuScenes root (contains samples/, sweeps/, v1.0-*/).")
    ap.add_argument("--version", default="v1.0-trainval", help="nuScenes version, e.g. v1.0-trainval or v1.0-mini")
    ap.add_argument("--seconds", type=float, default=4.0, help="Required past duration in seconds (keyframes are ~2Hz).")
    ap.add_argument("--n", type=int, default=10, help="How many tokens to print.")
    args = ap.parse_args()

    # Lazy import to avoid requiring devkit before args are parsed
    from nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    def has_past(sample_token: str, seconds: float) -> bool:
        """Return True if we can walk 'prev' links back >= `seconds` (keyframes)."""
        s = nusc.get("sample", sample_token)
        t0 = s["timestamp"]  # microseconds
        while s.get("prev"):
            s = nusc.get("sample", s["prev"])
            if (t0 - s["timestamp"]) / 1e6 >= seconds:
                return True
        return False

    count = 0
    for s in nusc.sample:
        tok = s["token"]
        if has_past(tok, args.seconds):
            scene = nusc.get("scene", s["scene_token"])
            log = nusc.get("log", scene["log_token"])
            print(f"{tok}  | scene={scene['name']} | location={log['location']}")
            count += 1
            if count >= args.n:
                break

    if count == 0:
        print(f"[Info] No samples with ≥{args.seconds:.1f}s of past found. "
              f"Try lowering --seconds or switch --version.")

if __name__ == "__main__":
    main()