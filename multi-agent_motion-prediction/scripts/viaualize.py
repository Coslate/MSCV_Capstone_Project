# scripts/visualize.py
import os, argparse, shutil, subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from dataset.packed_nuscenes_dataset import PackedNuScenesDataset, packed_nuscenes_collate
from models.baselines import GRUSingleRollout, GRUMultiRollout

def _color(i):  # for agents
    import matplotlib.pyplot as _plt
    return _plt.get_cmap("tab20")(i % 20)[:3]

def draw_frame(ax, item, pred, mode_desc=""):
    # map layers
    for k, style in [
        ("map_lane_center", dict(lw=0.9, a=0.6, c=(0.2,0.5,0.9))),
        ("map_lane_divider", dict(lw=0.8, a=0.4, c=(0.5,0.5,0.5))),
        ("map_road_divider", dict(lw=1.2, a=0.5, c=(0.25,0.25,0.25))),
        ("map_stop_line",    dict(lw=1.4, a=0.7, c=(0.8,0.1,0.1))),
    ]:
        if k in item:
            for seg in item[k]:
                ax.plot(seg[:,0], seg[:,1], linewidth=style["lw"], alpha=style["a"], color=style["c"])
    if "map_ped_crossing" in item:
        for poly in item["map_ped_crossing"]:
            if poly.shape[0] >= 3:
                ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=0.12,
                                        edgecolor=(0.0,0.6,0.0), facecolor=(0.0,0.8,0.0)))

    ego_h = item["ego_hist_xy"]; ego_f = item["ego_fut_xy"]
    ax.plot(ego_h[:,0], ego_h[:,1], '--', color=(0.1,0.6,0.9), lw=2, label="ego past")
    ax.plot(ego_f[:,0], ego_f[:,1], '-',  color=(0.1,0.6,0.9), lw=2, label="ego fut GT")

    A_h = item["agents_hist_xy"]   # (Na,Th,2)
    A_f = item["agents_fut_xy"]    # (Na,Tf,2)
    Na  = A_h.shape[0]

    # pred shape: (Na,Tf,2)  — 已經選好 best-of-K 或單軌
    for a in range(Na):
        col = _color(a)
        ax.plot(A_h[a,:,0], A_h[a,:,1], '--', color=col, lw=1.8, label=None if a else "agent past")
        ax.plot(A_f[a,:,0], A_f[a,:,1], '-',  color=col, lw=1.8, label=None if a else "agent fut GT")
        ax.plot(pred[a,:,0], pred[a,:,1], '-', color="k", lw=2.2, label=None if a else "pred")

    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.set_xlabel("X (m) — ego@t0"); ax.set_ylabel("Y (m) — ego@t0")
    ax.set_title(mode_desc)
    ax.legend(fontsize=7, loc="upper right")

def infer_pred_for_item(model, device, item, multirollout=False):
    batch1 = packed_nuscenes_collate([item])
    ag_h = batch1["agents_hist_xy"].to(device)
    ag_hm= batch1["agents_hist_mask"].to(device)
    eg_h = batch1["ego_hist_xy"].to(device)
    eg_hm= batch1["ego_hist_mask"].to(device)
    gt   = batch1["agents_fut_xy"].to(device)
    gtm  = batch1["agents_fut_mask"].to(device)

    with torch.no_grad():
        if not multirollout:
            pred = model(ag_h, ag_hm, eg_h, eg_hm)          # (1,Na,Tf,2)
            pred = pred[0].cpu().numpy()                    # (Na,Tf,2)
        else:
            pred_k, _ = model(ag_h, ag_hm, eg_h, eg_hm)     # (1,Na,K,Tf,2)
            # best-of-K by ADE per agent
            diff = torch.linalg.norm(pred_k - gt.unsqueeze(2), dim=-1)  # (1,Na,K,Tf)
            counts = gtm.sum(-1).clamp(min=1).unsqueeze(2)             # (1,Na,1)
            ade_bk = (diff * gtm.unsqueeze(2)).sum(-1) / counts        # (1,Na,K)
            best = ade_bk[0].argmin(dim=1)                              # (Na,)
            Na, K = best.shape[0], pred_k.shape[2]
            pick = []
            for a in range(Na):
                pick.append(pred_k[0, a, best[a]].cpu().numpy())
            pred = np.stack(pick, axis=0)                               # (Na,Tf,2)
    return pred

def save_video_if_requested(out_dir, fps, out_mp4):
    if shutil.which("ffmpeg") is None: return
    # 建一個 list.txt
    txt = os.path.join(out_dir, "list.txt")
    with open(txt, "w") as f:
        for name in sorted(os.listdir(out_dir)):
            if name.lower().endswith(".png"):
                f.write(f"file '{name}'\n")
                f.write("duration {:.6f}\n".format(1.0/float(fps)))
    cmd = [
        "ffmpeg","-y","-r",str(fps),"-f","concat","-safe","0","-i",txt,
        "-pix_fmt","yuv420p","-vcodec","libx264", out_mp4
    ]
    try:
        subprocess.run(cmd, cwd=out_dir, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[video] saved -> {out_mp4}")
    except Exception:
        print("[video] ffmpeg failed; frames are still saved.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--multirollout", action="store_true")
    ap.add_argument("--K", type=int, default=6)

    ap.add_argument("--include_map", action="store_true")
    ap.add_argument("--out_dir", required=True)

    # mode 選擇
    sub = ap.add_subparsers(dest="mode", required=True)

    # single
    sp_single = sub.add_parser("single", help="visualize a single sample")
    g = sp_single.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int)
    g.add_argument("--sample_token", type=str)
    sp_single.add_argument("--agent", type=int, default=None,
                           help="if set, only draw this agent; else draw all agents")

    # sequence
    sp_seq = sub.add_parser("sequence", help="visualize a consecutive sequence")
    sp_seq.add_argument("--start_idx", type=int, required=True)
    sp_seq.add_argument("--num", type=int, default=60)
    sp_seq.add_argument("--ensure_same_scene", action="store_true")
    sp_seq.add_argument("--fps", type=int, default=10)
    sp_seq.add_argument("--make_video", action="store_true")

    args = ap.parse_args()

    ds = PackedNuScenesDataset(args.manifest, include_map=args.include_map)
    Th, Tf = ds.Th, ds.Tf
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.multirollout:
        model = GRUSingleRollout(Th=Th, Tf=Tf).to(device)
    else:
        model = GRUMultiRollout(Th=Th, Tf=Tf, K=args.K).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        # 找到 index
        if args.idx is not None:
            idx = args.idx
        else:
            # linear scan：manifest 不大時可接受；若很大，可用事先建立 token->idx map
            idx = None
            for i, row in enumerate(ds.rows):
                if row["sample_token"] == args.sample_token:
                    idx = i; break
            if idx is None:
                raise ValueError("sample_token not found in manifest.")
        item = ds[idx]
        pred = infer_pred_for_item(model, device, item, multirollout=args.multirollout)

        # 只畫單一 agent？
        if args.agent is not None:
            # 改成只畫該 agent（其他 agent 當成空）
            Na = pred.shape[0]
            keep = args.agent
            # 造一個只有該 agent 的視覺化 item
            item_vis = {
                "ego_hist_xy": item["ego_hist_xy"],
                "ego_fut_xy":  item["ego_fut_xy"],
                "agents_hist_xy": item["agents_hist_xy"][keep:keep+1],
                "agents_fut_xy":  item["agents_fut_xy"][keep:keep+1],
            }
            if args.include_map:
                for k in ["map_lane_center","map_lane_divider","map_road_divider","map_ped_crossing","map_stop_line"]:
                    if k in item: item_vis[k] = item[k]
            fig, ax = plt.subplots(1,1, figsize=(6,6))
            draw_frame(ax, item_vis, pred[keep:keep+1], mode_desc=f"single idx={idx} agent={keep}")
            out_png = os.path.join(args.out_dir, f"single_idx{idx}_agent{keep}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[save] {out_png}")
        else:
            # 畫全部 agents
            vis_item = {
                "ego_hist_xy": item["ego_hist_xy"],
                "ego_fut_xy":  item["ego_fut_xy"],
                "agents_hist_xy": item["agents_hist_xy"],
                "agents_fut_xy":  item["agents_fut_xy"],
            }
            if args.include_map:
                for k in ["map_lane_center","map_lane_divider","map_road_divider","map_ped_crossing","map_stop_line"]:
                    if k in item: vis_item[k] = item[k]
            fig, ax = plt.subplots(1,1, figsize=(6,6))
            draw_frame(ax, vis_item, pred, mode_desc=f"single idx={idx} (all agents)")
            out_png = os.path.join(args.out_dir, f"single_idx{idx}_all.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[save] {out_png}")

    elif args.mode == "sequence":
        start = args.start_idx; num = args.num
        scene0 = ds.rows[start]["scene_token"]
        for t in range(num):
            i = start + t
            if i >= len(ds): break
            row = ds.rows[i]
            if args.ensure_same_scene and (row["scene_token"] != scene0):
                print(f"[stop] scene changed at idx={i}")
                break
            item = ds[i]
            pred = infer_pred_for_item(model, device, item, multirollout=args.multirollout)
            vis_item = {
                "ego_hist_xy": item["ego_hist_xy"],
                "ego_fut_xy":  item["ego_fut_xy"],
                "agents_hist_xy": item["agents_hist_xy"],
                "agents_fut_xy":  item["agents_fut_xy"],
            }
            if args.include_map:
                for k in ["map_lane_center","map_lane_divider","map_road_divider","map_ped_crossing","map_stop_line"]:
                    if k in item: vis_item[k] = item[k]
            fig, ax = plt.subplots(1,1, figsize=(6,6))
            draw_frame(ax, vis_item, pred, mode_desc=f"seq idx={i} t={t}")
            out_png = os.path.join(args.out_dir, f"frame_{t:05d}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[frames] saved to {args.out_dir}")

        if args.make_video:
            out_mp4 = os.path.join(args.out_dir, "sequence.mp4")
            save_video_if_requested(args.out_dir, fps=args.fps, out_mp4=out_mp4)

if __name__ == "__main__":
    main()
