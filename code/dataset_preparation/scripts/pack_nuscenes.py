#!/usr/bin/env python3
import os, json, argparse, numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
from tqdm import tqdm as _tqdm

from trmp.datasets.nuscenes import load_sample
from trmp.utils.map_helper import extract_map_vectors

def unicode_dtype_for(strings, min_U=16, margin=16):
    """給一串字，回傳足夠裝下它們的 dtype, 例如 'U312'。
    min_U: 最小寬度下限; margin: 多留一點空間避免剛好卡到邊界。
    """
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
    pad = np.full((T - t,) + a.shape[1:], fill, dtype=a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32)
    out = np.concatenate([a, pad], axis=0)
    m = np.zeros((T,), bool); m[:t] = True
    return out, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--past_sec", type=float, default=4.0)
    ap.add_argument("--future_sec", type=float, default=6.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    ap.add_argument("--map_radius_m", type=float, default=80.0)
    ap.add_argument("--keep_prefix", nargs="*", default=["vehicle.", "human.pedestrian"])
    ap.add_argument("--min_future_sec", type=float, default=0.0)
    ap.add_argument("--compress", action="store_true",
                help="Use np.savez_compressed (default: off)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man_fp = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")
    drop_fp = open(out_dir / "dropped.jsonl", "a", encoding="utf-8")  # 追加，便於多次執行對照

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    nmap_cache = {}
    stride_k = max(1, int(round(args.stride_sec * 2.0)))  # keyframe=2Hz

    total = kept = 0
    for scene in tqdm(nusc.scene, desc="Scenes", dynamic_ncols=True):
        # gather scene's keyframes
        toks = []
        s = nusc.get("sample", scene["first_sample_token"])
        while True:
            toks.append(s["token"])
            if not s["next"]: break
            s = nusc.get("sample", s["next"])

        samps = toks[::stride_k]

        # 內層：此 scene 的 keyframes 進度條
        pbar = tqdm(samps,
                    desc=f"{scene.get('name', scene['token'])} | keyframes",
                    leave=False,
                    dynamic_ncols=True)

        for samp_tok in pbar:
            total += 1
            idx = total - 1  # 這筆樣本的全域流水號（從 0 開始），無論保留或丟棄都會遞增
            data = load_sample(
                args.dataroot, samp_tok,
                past_sec=args.past_sec,
                future_sec=args.future_sec,
                map_radius_m=args.map_radius_m,
                version=args.version,
                nusc=nusc,                 # <-- 重用
                nmap_cache=nmap_cache      # <-- 重用
            )

            # transforms & convenience
            R_we, t_we = data["t0"]["ego_rot"], data["t0"]["ego_trans"]
            to_ego = world_to_ego_xy_fn(R_we, t_we)
            hist_hz = float(data["timestamps"].get("history_hz", 2.0))

            # ego yaw0 in world at t0
            eh = data["ego_history"]; yaw0_world = eh[-1, 2] if (isinstance(eh, np.ndarray) and eh.size) else 0.0

            # --- ego (to ego@t0) ---
            ego_hist_xy = to_ego(eh[:, :2]) if eh.size else np.zeros((0,2), np.float32)
            ego_hist_yaw = wrap_pi(eh[:, 2] - yaw0_world) if eh.size else np.zeros((0,), np.float32)

            ef = data.get("ego_future", np.zeros((0,3), np.float32))
            ego_fut_xy  = to_ego(ef[:, :2]) if ef.size else np.zeros((0,2), np.float32)
            ego_fut_yaw = wrap_pi(ef[:, 2] - yaw0_world) if ef.size else np.zeros((0,), np.float32)

            Th = int(round(args.past_sec * hist_hz)) + 1
            Tf = int(round(args.future_sec * hist_hz))

            ego_hist_xy, ego_hist_m = pad_to_len(ego_hist_xy, Th)
            ego_hist_yaw, _         = pad_to_len(ego_hist_yaw, Th)
            ego_fut_xy,  ego_fut_m  = pad_to_len(ego_fut_xy,  Tf)
            ego_fut_yaw, _          = pad_to_len(ego_fut_yaw, Tf)

            # --- agents ---
            keep_prefix = tuple(args.keep_prefix) if args.keep_prefix else None
            A_xy_h, A_yaw_h, A_m_h, A_xy_f, A_yaw_f, A_m_f, A_types, A_ids = ([], [], [], [], [], [], [], [])
            for aid, rec in data.get("agents_history", {}).items():
                typ = rec.get("type","")
                if keep_prefix and (not typ.startswith(keep_prefix)): continue

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

            need = int(round(args.min_future_sec * hist_hz))
            have = int(ego_fut_m.sum())
            if args.min_future_sec > 0 and have < need:
                # 記錄丟棄原因
                drop_fp.write(json.dumps({
                    "idx": int(idx),
                    "sample_token": samp_tok,
                    "scene_token": scene["token"],
                    "hist_hz": float(hist_hz),
                    "Tf_need": int(need),
                    "Tf_have": int(have),
                    "reason": "ego_future_too_short"
                }) + "\n")
                pbar.set_postfix(kept=kept, total=total, refresh=False)
                continue

            # --- map vectors（ego@t0） ---
            loc = data["city_or_map_id"]
            if loc not in nmap_cache:
                nmap_cache[loc] = NuScenesMap(dataroot=args.dataroot, map_name=loc)
            nmap = nmap_cache[loc]
            map_vec = extract_map_vectors(nmap, data["map"], to_ego)

            # --- I/O paths ---
            cam_names, cam_paths = [], []
            for cam, p in data.get("images", {}).items():
                cam_names.append(cam); cam_paths.append(p)
            #cam_names = np.array(cam_names, dtype=object)
            #cam_paths = np.array(cam_paths, dtype=object)
            #lidar_path = np.array(data["t0"]["lidar_filename"], dtype=object)

            # 相機名稱：通常短一點
            cam_names = np.array(
                cam_names,
                dtype=unicode_dtype_for(cam_names, min_U=16, margin=8)
            )

            # 圖像/點雲路徑：可能很長，寬鬆一點
            cam_paths = np.array(
                cam_paths,
                dtype=unicode_dtype_for(cam_paths, min_U=128, margin=128)
            )

            # lidar_path 建議存成 0 維 scalar（讀取時用 .item() 取出）
            lidar_path = np.array(
                data["t0"]["lidar_filename"],
                dtype=unicode_dtype_for(data["t0"]["lidar_filename"], min_U=128, margin=128)
            )

            # location 一般很短（"boston-seaport" / "singapore-hollandv" 等）
            location = np.array(
                loc,
                dtype=unicode_dtype_for(loc, min_U=16, margin=8)
            )            

            out_path = Path(args.out_dir) / f"{idx:08d}_{samp_tok}.npz"
            saver = np.savez_compressed if args.compress else np.savez
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
                scene_token=np.array(scene["token"], dtype=object),
                R_we=np.asarray(R_we, np.float32), t_we=np.asarray(t_we, np.float32),
                hist_hz=np.array(hist_hz, np.float32),
                past_sec=np.array(args.past_sec, np.float32),
                future_sec=np.array(args.future_sec, np.float32),
                stride_sec=np.array(args.stride_sec, np.float32),
                map_radius_m=np.array(args.map_radius_m, np.float32),
            )

            man_fp.write(json.dumps({
                "idx": int(idx),
                "npz": str(out_path),
                "sample_token": samp_tok,
                "scene_token": scene["token"],
                "location": loc,
                "Th": int(Th), "Tf": int(Tf),
                "Na": int(A_xy_h.shape[0]),
                "n_lanes": int(len(map_vec["lane_center"])),
                "n_stop": int(len(map_vec["stop_line"]))
            }) + "\n")

            # 寫 .npz、manifest 之後
            kept += 1

            # 右側摘要顯示目前累積
            pbar.set_postfix(kept=kept, total=total, refresh=False)            

    # 用 tqdm.write 避免把進度列沖掉
    man_fp.close()
    drop_fp.close()
    tqdm.write(f"[DONE] kept {kept}/{total} keyframes. Manifest -> {out_dir/'manifest.jsonl'}")

if __name__ == "__main__":
    main()
