# scripts/eval.py
import argparse, json, csv, os, shutil, subprocess
import torch
from torch.utils.data import DataLoader
from dataset.packed_nuscenes_dataset import PackedNuScenesDataset, packed_nuscenes_collate
from models.baselines import GRUSingleRollout, GRUMultiRollout
from metrics.trajectory_metrics import ade_fde_single, metrics_multirollout
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def _color_for_key(k: str):
    import matplotlib.pyplot as _plt
    return _plt.get_cmap("tab20")(hash(str(k)) % 20)[:3]

def _plot_one(ax, item, pred_agent_xy, agent_id, title=""):
    # map (若有)
    if "map_lane_center" in item:
        for seg in item["map_lane_center"]:
            ax.plot(seg[:,0], seg[:,1], linewidth=0.9, alpha=0.6, color=(0.2,0.5,0.9), label=None)
    if "map_lane_divider" in item:
        for seg in item["map_lane_divider"]:
            ax.plot(seg[:,0], seg[:,1], linewidth=0.8, alpha=0.4, color=(0.5,0.5,0.5), label=None)
    if "map_road_divider" in item:
        for seg in item["map_road_divider"]:
            ax.plot(seg[:,0], seg[:,1], linewidth=1.2, alpha=0.5, color=(0.25,0.25,0.25), label=None)
    if "map_ped_crossing" in item:
        for poly in item["map_ped_crossing"]:
            if poly.shape[0] >= 3:
                ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=0.12,
                                        edgecolor=(0.0,0.6,0.0), facecolor=(0.0,0.8,0.0)))
    if "map_stop_line" in item:
        for seg in item["map_stop_line"]:
            ax.plot(seg[:,0], seg[:,1], linewidth=1.4, alpha=0.7, color=(0.8,0.1,0.1), label=None)

    # ego
    ego_h = item["ego_hist_xy"]; ego_f = item["ego_fut_xy"]
    ax.plot(ego_h[:,0], ego_h[:,1], '--', color=(0.1,0.6,0.9), linewidth=2.0, label="ego past")
    ax.plot(ego_f[:,0], ego_f[:,1], '-',  color=(0.1,0.6,0.9), linewidth=2.0, label="ego fut(GT)")

    # agent
    A_h = item["agents_hist_xy"][agent_id]
    A_f = item["agents_fut_xy"][agent_id]
    col = _color_for_key(agent_id)
    ax.plot(A_h[:,0], A_h[:,1], '--', color=col, linewidth=2.0, label=f"agent[{agent_id}] past")
    ax.plot(A_f[:,0], A_f[:,1], '-',  color=col, linewidth=2.0, label=f"agent[{agent_id}] fut(GT)")
    ax.plot(pred_agent_xy[:,0], pred_agent_xy[:,1], '-', color='k', linewidth=2.2, label="pred")

    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.set_xlabel("X (m) — ego@t0"); ax.set_ylabel("Y (m) — ego@t0")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--multirollout", action="store_true")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--miss_thresh", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")

    # 新增：輸出 per-agent CSV
    ap.add_argument("--dump_csv", type=str, default=None,
                    help="write per-agent metrics to CSV")

    # 新增：直接在 eval 階段存圖（前 K 張『最難』樣本）
    ap.add_argument("--save_vis_dir", type=str, default=None,
                    help="if set, save visualizations to this dir")
    ap.add_argument("--topk_vis", type=int, default=50)

    # 排序依據（單軌用 ADE/FDE，多軌常用 minADE_K/minFDE_K）
    ap.add_argument("--rank_by", type=str, default="ADE",
                    choices=["ADE","FDE","minADE_K","minFDE_K"])

    # 畫圖時是否把地圖 read 進 dataset
    ap.add_argument("--vis_with_map", action="store_true",
                    help="when saving vis, re-open dataset with include_map=True for plotting map layers")

    args = ap.parse_args()

    ds = PackedNuScenesDataset(args.manifest, include_map=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=packed_nuscenes_collate, pin_memory=True)
    Th, Tf = ds.Th, ds.Tf

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.multirollout:
        model = GRUSingleRollout(Th=Th, Tf=Tf).to(device)
    else:
        model = GRUMultiRollout(Th=Th, Tf=Tf, K=args.K).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    agg = {}
    rows = []  # CSV rows
    want_vis = args.save_vis_dir is not None
    if want_vis:
        os.makedirs(args.save_vis_dir, exist_ok=True)

    # 暫存 (score, idx, agent_id, token) for topK
    pool = []

    with torch.no_grad():
        for batch in dl:
            ag_h = batch["agents_hist_xy"].to(device)
            ag_hm= batch["agents_hist_mask"].to(device)
            eg_h = batch["ego_hist_xy"].to(device)
            eg_hm= batch["ego_hist_mask"].to(device)
            gt   = batch["agents_fut_xy"].to(device)
            gtm  = batch["agents_fut_mask"].to(device)
            meta = batch["meta"]
            B, Na, _, _ = gt.shape

            if not args.multirollout:
                pred = model(ag_h, ag_hm, eg_h, eg_hm)  # (B,Na,Tf,2)
                m = ade_fde_single(pred, gt, gtm)
                for k,v in m.items():
                    if isinstance(v, torch.Tensor) and v.ndim==0:
                        agg[k] = agg.get(k, 0.0) + float(v)

                ade_pa = m["per_agent_ADE"].cpu()   # (B,Na)
                fde_pa = m["per_agent_FDE"].cpu()   # (B,Na)
                has_pa = m["per_agent_has"].cpu()   # (B,Na) bool
                steps  = gtm.sum(dim=-1).cpu()

                for b in range(B):
                    idx = int(meta["idx"][b]); token = meta["sample_token"][b]
                    for a in range(Na):
                        if not bool(has_pa[b,a]):
                            continue
                        row = {
                            "idx": idx,
                            "sample_token": token,
                            "agent": a,
                            "steps_valid": int(steps[b,a].item()),
                            "ADE": float(ade_pa[b,a].item()),
                            "FDE": float(fde_pa[b,a].item()),
                        }
                        rows.append(row)
                        if want_vis:
                            score = row[args.rank_by]  # ADE or FDE
                            pool.append((score, idx, a, token))

            else:
                pred_k, _ = model(ag_h, ag_hm, eg_h, eg_hm)  # (B,Na,K,Tf,2)
                m1 = ade_fde_single(pred_k[:,:,0], gt, gtm) # 參考單條
                for k,v in m1.items():
                    if isinstance(v, torch.Tensor) and v.ndim==0:
                        agg["single_"+k] = agg.get("single_"+k, 0.0) + float(v)

                m2 = metrics_multirollout(pred_k, gt, gtm, miss_thresh=args.miss_thresh)
                for k,v in m2.items():
                    if isinstance(v, torch.Tensor) and v.ndim==0:
                        agg[k] = agg.get(k, 0.0) + float(v)

                ade_pa = m1["per_agent_ADE"].cpu()
                fde_pa = m1["per_agent_FDE"].cpu()
                has_pa = m1["per_agent_has"].cpu()
                steps  = gtm.sum(dim=-1).cpu()
                minADE_pa = m2["per_agent_minADE"].cpu()
                minFDE_pa = m2["per_agent_minFDE"].cpu()

                for b in range(B):
                    idx = int(meta["idx"][b]); token = meta["sample_token"][b]
                    for a in range(Na):
                        if not bool(has_pa[b,a]):
                            continue
                        row = {
                            "idx": idx,
                            "sample_token": token,
                            "agent": a,
                            "steps_valid": int(steps[b,a].item()),
                            "ADE": float(ade_pa[b,a].item()),
                            "FDE": float(fde_pa[b,a].item()),
                            "minADE_K": float(minADE_pa[b,a].item()),
                            "minFDE_K": float(minFDE_pa[b,a].item()),
                        }
                        rows.append(row)
                        if want_vis:
                            key = args.rank_by
                            score = row[key] if key in row else row["ADE"]
                            pool.append((score, idx, a, token))

    # 平均
    n = len(dl)
    for k in list(agg.keys()):
        agg[k] /= max(n, 1)
    print(json.dumps(agg, indent=2))

    # CSV
    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)
        fields = sorted(set().union(*[r.keys() for r in rows])) if rows else ["idx","sample_token","agent","steps_valid","ADE","FDE"]
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[write] per-agent CSV -> {args.dump_csv} ({len(rows)} rows)")

    # VIS top-K
    if want_vis and rows:
        pool.sort(key=lambda x: x[0], reverse=True)
        top = pool[: args.topk_vis]
        # 重新打開 dataset(include_map=...) 以畫圖
        ds_map = PackedNuScenesDataset(args.manifest, include_map=args.vis_with_map)
        with torch.no_grad():
            for rank, (score, idx, agent_id, token) in enumerate(top, 1):
                item = ds_map[idx]
                batch1 = packed_nuscenes_collate([item])
                ag_h = batch1["agents_hist_xy"].to(device)
                ag_hm= batch1["agents_hist_mask"].to(device)
                eg_h = batch1["ego_hist_xy"].to(device)
                eg_hm= batch1["ego_hist_mask"].to(device)
                gt   = batch1["agents_fut_xy"].to(device)
                gtm  = batch1["agents_fut_mask"].to(device)

                if not args.multirollout:
                    pred = model(ag_h, ag_hm, eg_h, eg_hm)          # (1,Na,Tf,2)
                    pred_agent = pred[0, agent_id].cpu().numpy()
                else:
                    pred_k, _ = model(ag_h, ag_hm, eg_h, eg_hm)     # (1,Na,K,Tf,2)
                    # best-of-K by ADE
                    diff = torch.linalg.norm(pred_k - gt.unsqueeze(2), dim=-1)  # (1,Na,K,Tf)
                    counts = gtm.sum(-1).clamp(min=1).unsqueeze(2)            # (1,Na,1)
                    ade_bk = (diff * gtm.unsqueeze(2)).sum(-1) / counts      # (1,Na,K)
                    best_k = ade_bk[0, agent_id].argmin().item()
                    pred_agent = pred_k[0, agent_id, best_k].cpu().numpy()

                vis_item = {
                    "ego_hist_xy": batch1["ego_hist_xy"][0].cpu().numpy(),
                    "ego_fut_xy":  batch1["ego_fut_xy"][0].cpu().numpy(),
                    "agents_hist_xy": batch1["agents_hist_xy"][0].cpu().numpy(),
                    "agents_fut_xy":  batch1["agents_fut_xy"][0].cpu().numpy(),
                }
                if args.vis_with_map:
                    for k in ["map_lane_center","map_lane_divider","map_road_divider",
                              "map_ped_crossing","map_stop_line"]:
                        if k in item: vis_item[k] = item[k]

                fig, ax = plt.subplots(1,1, figsize=(6,6))
                _plot_one(ax, vis_item, pred_agent, agent_id,
                          title=f"{args.rank_by}={score:.2f} | idx={idx} a={agent_id}")
                out_png = os.path.join(args.save_vis_dir, f"rank{rank:03d}_{args.rank_by}_{score:.2f}_idx{idx}_a{agent_id}.png")
                plt.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"[vis] saved top-{args.topk_vis} images to {args.save_vis_dir}")

if __name__ == "__main__":
    main()
