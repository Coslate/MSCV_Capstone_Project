# scripts/make_splits.py
import json, random, argparse, os
from collections import defaultdict
from pathlib import Path

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--test_ratio",  type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    # 讀入所有列
    rows = [json.loads(l) for l in open(args.manifest, "r", encoding="utf-8") if l.strip()]
    # 依 scene 分組（確保同一 scene 不會跨 split）
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene_token"]].append(r)

    scenes = list(by_scene.keys())
    random.Random(args.seed).shuffle(scenes)

    total = sum(len(by_scene[s]) for s in scenes)
    tgt_train = int(total * args.train_ratio)
    tgt_val   = int(total * args.val_ratio)
    tgt_test  = total - tgt_train - tgt_val

    train, val, test = [], [], []
    c_train = c_val = c_test = 0

    # 盡量按目標數量分配整個 scene
    for s in scenes:
        block = by_scene[s]
        if c_train + len(block) <= tgt_train:
            train += block; c_train += len(block)
        elif c_val + len(block) <= tgt_val:
            val += block; c_val += len(block)
        else:
            test += block; c_test += len(block)

    # 輸出
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p_train = out_dir / "manifest.train.jsonl"
    p_val   = out_dir / "manifest.val.jsonl"
    p_test  = out_dir / "manifest.test.jsonl"
    write_jsonl(p_train, train)
    write_jsonl(p_val,   val)
    write_jsonl(p_test,  test)

    # 摘要
    print(f"[done] total={total}  scenes={len(scenes)} | "
          f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"-> {p_train}\n-> {p_val}\n-> {p_test}")

if __name__ == "__main__":
    main()
