#!/usr/bin/env python3
import os, json, argparse, numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes import NuScenes
from pathlib import Path

# ==== project imports ====
from trmp.datasets.nuscenes import load_sample
from trmp.utils.map_helper import extract_map_vectors

# --------- helpers ----------
def unicode_dtype_for(strings, min_U=16, margin=16):
    if isinstance(strings, str):
        maxlen = len(strings)
    else:
        strings = [s if isinstance(s, str) else str(s) for s in list(strings)]
        maxlen = max([len(s) for s in strings], default=0)
    width = max(maxlen + margin, min_U)
    return f"U{width}"

def wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi

def world_to_ego_xy_fn(R_we, t_we):
    R2 = np.asarray(R_we, dtype=np.float32)[:2, :2]
    tw = np.asarray(t_we, dtype=np.float32)[:2]
    def f(xy_world):
        if xy_world is None: return None
        xy = np.asarray(xy_world, dtype=np.float32)
        xy = np.atleast_2d(xy)[:, :2]
        return (xy - tw) @ R2
    return f

def pad_to_len(a, T, fill=np.nan):
    a = np.asarray(a) if a is not None else np.zeros((0,), np.float32)
    if a.ndim == 1: a = a[:, None]
    t = a.shape[0]
    if t >= T: return a[:T], np.ones((T,), bool)
    pad = np.full((T - t,) + a.shape[1:], fill,
                  dtype=a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32)
    out = np.concatenate([a, pad], axis=0)
    m = np.zeros((T,), bool); m[:t] = True
    return out, m

# ====== worker (平行時用；會在每個 process 內初始化一次 NuScenes/Map) ======
_GLOBALS = {"nusc": None, "version": None, "dataroot": None, "nmap_cache": {}}

def _worker_init(dataroot, version):
    _GLOBALS["dataroot"] = dataroot
    _GLOBALS["version"] = version
    _GLOBALS["nusc"] = NuScenes(version=version, dataroot=dataroot, verbose=False)
    _GLOBALS["nmap_cache"] = {}   # per-process cache

def _pack_one_token(job):
    """
    job: dict with keys:
      idx, sample_token, past_sec, future_sec, map_radius_m, keep_prefix, compress
    returns: (idx, kept, manifest_record or None)
    """
    idx = job["idx"]
    dataroot = _GLOBALS["dataroot"]
    nusc = _GLOBALS["nusc"]
    nmap_cache = _GLOBALS["nmap_cache"]
    scene_tok = nusc.get('sample', job['sample_token'])['scene_token']   # 先抓好，drop 時也能用

    # 讀一筆（這裡用 per-process 的 nusc；不會重複載表）
    data = load_sample(
        nuscenes_root=dataroot,
        sample_token=job["sample_token"],
        past_sec=job["past_sec"],
        future_sec=job["future_sec"],
        map_radius_m=job["map_radius_m"],
        nusc=nusc,
        nmap_cache=nmap_cache,   # 讓 load_sample 用同一份地圖快取
    )

    # transforms & convenience
    R_we, t_we = data["t0"]["ego_rot"], data["t0"]["ego_trans"]
    to_ego = world_to_ego_xy_fn(R_we, t_we)
    hist_hz = float(data["timestamps"].get("history_hz", 2.0))

    # ego yaw0 in world at t0
    eh = data["ego_history"]
    yaw0_world = eh[-1, 2] if (isinstance(eh, np.ndarray) and eh.size) else 0.0

    # --- ego ---
    ego_hist_xy  = to_ego(eh[:, :2]) if eh.size else np.zeros((0,2), np.float32)
    ego_hist_yaw = wrap_pi(eh[:, 2] - yaw0_world) if eh.size else np.zeros((0,), np.float32)
    ef = data.get("ego_future", np.zeros((0,3), np.float32))
    ego_fut_xy   = to_ego(ef[:, :2]) if ef.size else np.zeros((0,2), np.float32)
    ego_fut_yaw  = wrap_pi(ef[:, 2] - yaw0_world) if ef.size else np.zeros((0,), np.float32)

    Th = int(round(job["past_sec"] * hist_hz)) + 1
    Tf = int(round(job["future_sec"] * hist_hz))

    ego_hist_xy,  ego_hist_m = pad_to_len(ego_hist_xy,  Th)
    ego_hist_yaw, _          = pad_to_len(ego_hist_yaw, Th)
    ego_fut_xy,   ego_fut_m  = pad_to_len(ego_fut_xy,   Tf)
    ego_fut_yaw,  _          = pad_to_len(ego_fut_yaw,  Tf)

    # --- agents ---
    keep_prefix = tuple(job["keep_prefix"]) if job["keep_prefix"] else None
    A_xy_h, A_yaw_h, A_m_h, A_xy_f, A_yaw_f, A_m_f, A_types, A_ids = [[] for _ in range(8)]

    for aid, rec in data.get("agents_history", {}).items():
        typ = rec.get("type", "")
        if keep_prefix and (not typ.startswith(keep_prefix)): 
            continue

        xy_h, m_h, yaw_h = rec["xy"], rec["mask"], rec["yaw"]
        if isinstance(m_h, np.ndarray) and m_h.dtype==bool and m_h.shape[0]==xy_h.shape[0]:
            xy_h = xy_h[m_h]; yaw_h = yaw_h[m_h]
        xy_h = to_ego(xy_h)
        yaw_h = wrap_pi(yaw_h - yaw0_world)

        f = data.get("agents_future", {}).get(aid, {})
        xy_f, m_f, yaw_f = f.get("xy"), f.get("mask"), f.get("yaw")
        if isinstance(xy_f, np.ndarray) and xy_f.size:
            if isinstance(m_f, np.ndarray) and m_f.dtype==bool and m_f.shape[0]==xy_f.shape[0]:
                xy_f = xy_f[m_f]; yaw_f = yaw_f[m_f]
            xy_f = to_ego(xy_f)
            yaw_f = wrap_pi(yaw_f - yaw0_world)
        else:
            xy_f = np.zeros((0,2), np.float32); yaw_f = np.zeros((0,), np.float32)

        xy_h, m1 = pad_to_len(xy_h, Th)
        yaw_h,_  = pad_to_len(yaw_h, Th)
        xy_f, m2 = pad_to_len(xy_f, Tf)
        yaw_f,_  = pad_to_len(yaw_f, Tf)

        A_xy_h.append(xy_h); A_yaw_h.append(yaw_h); A_m_h.append(m1)
        A_xy_f.append(xy_f); A_yaw_f.append(yaw_f); A_m_f.append(m2)
        A_types.append(typ); A_ids.append(str(aid))

    A_xy_h = np.stack(A_xy_h, 0) if A_xy_h else np.zeros((0,Th,2), np.float32)
    A_yaw_h= np.stack(A_yaw_h,0) if A_yaw_h else np.zeros((0,Th),   np.float32)
    A_m_h  = np.stack(A_m_h,  0) if A_m_h   else np.zeros((0,Th),   bool)
    A_xy_f = np.stack(A_xy_f, 0) if A_xy_f else np.zeros((0,Tf,2), np.float32)
    A_yaw_f= np.stack(A_yaw_f,0) if A_yaw_f else np.zeros((0,Tf),   np.float32)
    A_m_f  = np.stack(A_m_f,  0) if A_m_f   else np.zeros((0,Tf),   bool)
    A_types= np.array(A_types, dtype=object)
    A_ids  = np.array(A_ids,   dtype=object)

    # 早退條件：未來太短要丟
    if job["min_future_sec"] > 0 and ego_fut_m.sum() < int(round(job["min_future_sec"]*hist_hz)):
        return idx, False, {
        "idx": int(idx),
        "sample_token": job["sample_token"],
        "scene_token": scene_tok,
        "Tf_have": int(ego_fut_m.sum()),
        "Tf_need": int(round(job["min_future_sec"]*hist_hz)),
        "hist_hz": float(hist_hz),
        "reason": "ego_future_too_short"
        }

    # --- map vectors（ego@t0） ---
    loc = data["city_or_map_id"]
    nmap_cache = _GLOBALS["nmap_cache"]
    if loc not in nmap_cache:
        nmap_cache[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
    nmap = nmap_cache[loc]
    map_vec = extract_map_vectors(nmap, data["map"], to_ego)

    # --- I/O fields (paths & meta) ---
    cam_names, cam_paths = [], []
    for cam, p in data.get("images", {}).items():
        cam_names.append(cam); cam_paths.append(p)
    cam_names = np.array(cam_names, dtype=unicode_dtype_for(cam_names, min_U=16, margin=8))
    cam_paths = np.array(cam_paths, dtype=unicode_dtype_for(cam_paths, min_U=128, margin=128))
    lidar_path = np.array(data["t0"]["lidar_filename"],
                          dtype=unicode_dtype_for(data["t0"]["lidar_filename"], min_U=128, margin=128))
    location = np.array(loc, dtype=unicode_dtype_for(loc, min_U=16, margin=8))

    # out file
    out_dir = Path(job["out_dir"])
    out_path = out_dir / f"{job['idx']:08d}_{job['sample_token']}.npz"
    saver = np.savez_compressed if job["compress"] else np.savez
    samp_tok = job["sample_token"]
    scene_tok = _GLOBALS["nusc"].get('sample', samp_tok)['scene_token']
    saver(
        out_path,
        # ego
        ego_hist_xy=ego_hist_xy, ego_hist_yaw=ego_hist_yaw, ego_hist_mask=ego_hist_m,
        ego_fut_xy=ego_fut_xy,   ego_fut_yaw=ego_fut_yaw,   ego_fut_mask=ego_fut_m,
        # agents
        agents_hist_xy=A_xy_h, agents_hist_yaw=A_yaw_h, agents_hist_mask=A_m_h,
        agents_fut_xy=A_xy_f,  agents_fut_yaw=A_yaw_f,  agents_fut_mask=A_m_f,
        agents_type=A_types, agents_id=A_ids,
        # map vectors
        map_lane_center=map_vec["lane_center"],
        map_lane_divider=map_vec["lane_divider"],
        map_road_divider=map_vec["road_divider"],
        map_ped_crossing=map_vec["ped_crossing"],
        map_stop_line=map_vec["stop_line"],
        map_traffic_light=map_vec["traffic_light"],
        # files & meta
        cam_names=cam_names, cam_paths=cam_paths, lidar_path=lidar_path,
        location=location,
        sample_token=np.array(samp_tok, dtype=object),
        scene_token=np.array(scene_tok, dtype=object),
        R_we=np.asarray(R_we, np.float32), t_we=np.asarray(t_we, np.float32),
        hist_hz=np.array(hist_hz, np.float32),
        past_sec=np.array(job["past_sec"], np.float32),
        future_sec=np.array(job["future_sec"], np.float32),
        stride_sec=np.array(job["stride_sec"], np.float32),
        map_radius_m=np.array(job["map_radius_m"], np.float32),
    )

    manifest_row = {
        "idx": int(idx),
        "npz": str(out_path),
        "sample_token": samp_tok,
        "scene_token": scene_tok,
        "location": loc,
        "Th": int(Th), "Tf": int(Tf),
        "Na": int(A_xy_h.shape[0]),
        "n_lanes": int(len(map_vec["lane_center"])),
        "n_stop": int(len(map_vec["stop_line"]))
    }
    return idx, True, manifest_row

# ====== main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--past_sec", type=float, default=4.0)
    ap.add_argument("--future_sec", type=float, default=6.0)
    ap.add_argument("--stride_sec", type=float, default=0.5, help="0.5s=每個 keyframe；1.0=每兩個")
    ap.add_argument("--map_radius_m", type=float, default=80.0)
    ap.add_argument("--keep_prefix", nargs="*", default=["vehicle.", "human.pedestrian"])
    ap.add_argument("--min_future_sec", type=float, default=0.0)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0=序列；>0=平行")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man_fp = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")

    # 先列出「所有樣本」並給流水號 idx（這個 idx 就是 manifest 順序）
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    stride_k = max(1, int(round(args.stride_sec * 2.0)))  # 2Hz keyframes
    jobs = []
    for sc_i, scene in enumerate(tqdm(nusc.scene, desc="Scenes")):
        toks = []
        s = nusc.get("sample", scene["first_sample_token"])
        while True:
            toks.append(s["token"])
            if not s["next"]: break
            s = nusc.get("sample", s["next"])
        for tok in toks[::stride_k]:
            jobs.append({
                "idx": len(jobs),
                "sample_token": tok,
                "past_sec": args.past_sec,
                "future_sec": args.future_sec,
                "map_radius_m": args.map_radius_m,
                "keep_prefix": args.keep_prefix,
                "min_future_sec": args.min_future_sec,
                "compress": args.compress,
                "out_dir": str(out_dir),
                "stride_sec": args.stride_sec,
            })

    kept = 0
    if args.workers <= 0:
        # --- 序列：用同一個 nusc，速度也不錯，且記憶體省 ---
        _GLOBALS["dataroot"] = args.dataroot
        _GLOBALS["version"]  = args.version
        _GLOBALS["nusc"]     = nusc
        _GLOBALS["nmap_cache"] = {}   # 一次性地圖快取        

        for job in tqdm(jobs, desc="Packing", total=len(jobs)):
            # 直接把本機 nusc 傳進 load_sample（避免重複載表）
            # 為了與 worker 版本共用邏輯，這裡偷懶呼叫 _worker_init 一次以設置全域 dataroot/version
            idx, ok, row = _pack_one_token(job)
            if ok:
                man_fp.write(json.dumps(row) + "\n")
                kept += 1
    else:
        # --- 平行：每個 process 初始化一次 NuScenes 與 Map cache ---
        next_to_write = 0
        buffer = {}          # idx -> row (保留尚未輪到寫的 keep)
        dropped = set()      # 記錄被丟掉的 idx

        dropped_fp = open(out_dir / "dropped.jsonl", "w", encoding="utf-8")

        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                         initargs=(args.dataroot, args.version)) as ex:
            futs = [ex.submit(_pack_one_token, job) for job in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Packing(parallel)"):
                idx, ok, row = fut.result()

                if ok:
                    buffer[idx] = row
                else:
                    # 記住這個 idx 被丟掉，並（最好）把原因寫進 dropped.jsonl
                    dropped.add(idx)
                    if row is not None:
                        dropped_fp.write(json.dumps(row) + "\n")

                # 嘗試把「從 next_to_write 開始的連續前綴」寫出去
                while True:
                    if next_to_write in dropped:
                        # 這個位置被丟掉，直接前進
                        next_to_write += 1
                        continue
                    if next_to_write in buffer:
                        man_fp.write(json.dumps(buffer.pop(next_to_write)) + "\n")
                        kept += 1
                        next_to_write += 1
                        continue
                    break

        dropped_fp.close()

    man_fp.close()
    print(f"[DONE] kept {kept}/{len(jobs)} keyframes. Manifest -> {out_dir/'manifest.jsonl'}")

if __name__ == "__main__":
    main()
