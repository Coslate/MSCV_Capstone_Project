#!/usr/bin/env python3
import argparse, json, random, numpy as np, os, sys

def load_manifest_row(manifest, idx=None, sample_token=None, random_pick=False):
    rows = [json.loads(l) for l in open(manifest, "r", encoding="utf-8") if l.strip()]
    if idx is not None:
        for r in rows:
            if int(r.get("idx", -1)) == int(idx):
                return r
        raise ValueError(f"idx={idx} not found.")
    if sample_token is not None:
        for r in rows:
            if r.get("sample_token") == sample_token:
                return r
        raise ValueError(f"sample_token={sample_token} not found.")
    if random_pick:
        return random.choice(rows)
    raise ValueError("Need --idx or --sample_token or --random.")

def _squeeze_last(a):
    """Make yaw arrays (Th,1) -> (Th,) / (Na,Th,1) -> (Na,Th)."""
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 1:
        return np.squeeze(a, axis=-1)
    return a

def _assert_same(msg, a, b):
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")

def _assert_true(msg, cond):
    if not cond:
        raise AssertionError(msg)

def _check_nan_mask(name, arr, mask):
    """
    arr: (..., T, D) or (..., T)
    mask: (..., T) bool
    Rule: finite only where mask==True; NaN allowed where mask==False.
    """
    arr = np.asarray(arr)
    mask = np.asarray(mask).astype(bool)
    # broadcast to (..., T)
    if arr.ndim == mask.ndim + 1:
        finite = np.all(np.isfinite(arr), axis=-1)   # all dims finite, e.g., D=2
    else:
        finite = np.isfinite(arr)
    _assert_true(f"{name}: mask shape mismatch", finite.shape == mask.shape)
    bad_true  = np.any(~finite &  mask)  # masked True but has NaN
    bad_false = np.any( finite & ~mask)  # masked False but still finite
    _assert_true(f"{name}: has NaN where mask=True", not bad_true)
    _assert_true(f"{name}: has finite values where mask=False", not bad_false)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--idx", type=int)
    g.add_argument("--sample_token")
    ap.add_argument("--random", action="store_true", help="pick a random row")
    args = ap.parse_args()

    row = load_manifest_row(args.manifest, idx=args.idx, sample_token=args.sample_token, random_pick=args.random)
    npz_path = row["npz"]
    print(f"[pick] idx={row['idx']}  token={row['sample_token']}  npz={npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    # manifest targets
    Th_m = int(row["Th"]); Tf_m = int(row["Tf"]); Na_m = int(row["Na"])

    # read arrays
    ego_xy_h  = np.asarray(d["ego_hist_xy"])
    ego_yaw_h = _squeeze_last(d["ego_hist_yaw"])
    ego_m_h   = np.asarray(d["ego_hist_mask"]).astype(bool)

    ego_xy_f  = np.asarray(d["ego_fut_xy"])
    ego_yaw_f = _squeeze_last(d["ego_fut_yaw"])
    ego_m_f   = np.asarray(d["ego_fut_mask"]).astype(bool)

    A_xy_h  = np.asarray(d["agents_hist_xy"])
    A_yaw_h = _squeeze_last(d["agents_hist_yaw"])
    A_m_h   = np.asarray(d["agents_hist_mask"]).astype(bool)

    A_xy_f  = np.asarray(d["agents_fut_xy"])
    A_yaw_f = _squeeze_last(d["agents_fut_yaw"])
    A_m_f   = np.asarray(d["agents_fut_mask"]).astype(bool)

    # ===== shape checks
    print("\n[SHAPE]")
    print(" ego_hist_xy:", ego_xy_h.shape, "  ego_fut_xy:", ego_xy_f.shape)
    print(" A_hist_xy:", A_xy_h.shape, "   A_fut_xy:", A_xy_f.shape)

    _assert_same("ego_hist_xy.shape", tuple(ego_xy_h.shape), (Th_m, 2))
    _assert_same("ego_fut_xy.shape", tuple(ego_xy_f.shape), (Tf_m, 2))

    _assert_same("agents_hist_xy.shape[0]=Na", int(A_xy_h.shape[0]), Na_m)
    _assert_same("agents_hist_xy.shape[1]=Th", int(A_xy_h.shape[1]), Th_m)
    _assert_same("agents_hist_xy.shape[2]=2",  int(A_xy_h.shape[2]), 2)

    _assert_same("agents_fut_xy.shape[0]=Na", int(A_xy_f.shape[0]), Na_m)
    _assert_same("agents_fut_xy.shape[1]=Tf", int(A_xy_f.shape[1]), Tf_m)
    _assert_same("agents_fut_xy.shape[2]=2",  int(A_xy_f.shape[2]), 2)

    # yaw shapes (after squeeze) should be (Th,) / (Na,Th)
    _assert_same("ego_hist_yaw.shape", tuple(ego_yaw_h.shape), (Th_m,))
    _assert_same("ego_fut_yaw.shape",  tuple(ego_yaw_f.shape), (Tf_m,))
    _assert_same("agents_hist_yaw.shape", A_yaw_h.shape, (Na_m, Th_m))
    _assert_same("agents_fut_yaw.shape",  A_yaw_f.shape, (Na_m, Tf_m))

    # mask shapes
    _assert_same("ego_hist_mask.shape", tuple(ego_m_h.shape), (Th_m,))
    _assert_same("ego_fut_mask.shape",  tuple(ego_m_f.shape), (Tf_m,))
    _assert_same("agents_hist_mask.shape", tuple(A_m_h.shape), (Na_m, Th_m))
    _assert_same("agents_fut_mask.shape",  tuple(A_m_f.shape), (Na_m, Tf_m))

    # ===== NaN <-> mask consistency
    print("\n[MASK vs NaN]")
    _check_nan_mask("ego_hist_xy",  ego_xy_h,  ego_m_h)
    _check_nan_mask("ego_hist_yaw", ego_yaw_h, ego_m_h)
    _check_nan_mask("ego_fut_xy",   ego_xy_f,  ego_m_f)
    _check_nan_mask("ego_fut_yaw",  ego_yaw_f, ego_m_f)

    _check_nan_mask("agents_hist_xy",  A_xy_h,  A_m_h)
    _check_nan_mask("agents_hist_yaw", A_yaw_h, A_m_h)
    _check_nan_mask("agents_fut_xy",   A_xy_f,  A_m_f)
    _check_nan_mask("agents_fut_yaw",  A_yaw_f, A_m_f)

    # ===== sanity: last hist step (t0) must be valid for ego
    # 允許右補齊：找出最後一個 True 的位置，視為 t0
    idxs = np.where(ego_m_h)[0]
    _assert_true("ego history has no valid steps", idxs.size > 0)
    last_true = int(idxs[-1])

    # mask 必須是「前綴 True、後綴 False」的連續形狀
    _assert_true(
        "ego_hist_mask must be contiguous (prefix True then False)",
        np.all(ego_m_h[:last_true+1]) and np.all(~ego_m_h[last_true+1:])
    )

    # t0 那一列必須是有限值
    _assert_true("ego_hist_xy at t0 should be finite", np.isfinite(ego_xy_h[last_true]).all())
    _assert_true("ego_hist_yaw at t0 should be finite", np.isfinite(ego_yaw_h[last_true]).all())


    print("\n✅ All checks passed.")

if __name__ == "__main__":
    main()
