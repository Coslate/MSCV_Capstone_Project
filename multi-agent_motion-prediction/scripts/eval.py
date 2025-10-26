# scripts/eval.py
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from datasets.packed_nuscenes_dataset import PackedNuScenesDataset, packed_nuscenes_collate
from models.baselines import GRUSingleRollout, GRUMultiRollout
from metrics.trajectory_metrics import ade_fde_single, metrics_multirollout

def _resolve_ckpt_path(p: str, prefer: str = "best.pt"):
    if p is None:
        return None
    if os.path.isdir(p):
        cand = os.path.join(p, prefer)
        return cand if os.path.isfile(cand) else None
    return p if os.path.isfile(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True, help="path to ckpt file or dir (will pick best.pt)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--multirollout", action="store_true")
    ap.add_argument("--K", type=int, default=None, help="override K if needed; default from ckpt")
    ap.add_argument("--miss_thresh", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", type=str, default=None, help="optional path to save metrics JSON")
    args = ap.parse_args()

    ckpt_path = _resolve_ckpt_path(args.ckpt, prefer="best.pt")
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ds = PackedNuScenesDataset(args.manifest)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=packed_nuscenes_collate, pin_memory=True)
    Th, Tf = ds.Th, ds.Tf

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- load ckpt & pick model hyperparams from it ---
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}

    hidden = int(ckpt_args.get("hidden", 128))
    num_layers = int(ckpt_args.get("num_layers", 1))
    d_model = int(ckpt_args.get("d_model", 128))
    K = args.K if (args.K is not None) else int(ckpt_args.get("K", 6))
    multirollout = bool(ckpt_args.get("multirollout", args.multirollout))

    if not multirollout:
        model = GRUSingleRollout(Th=Th, Tf=Tf, hidden=hidden, num_layers=num_layers, d_model=d_model).to(device)
    else:
        model = GRUMultiRollout(Th=Th, Tf=Tf, K=K, hidden=hidden, num_layers=num_layers, d_model=d_model).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- weighted aggregation: macro by #agents, micro by #points ---
    sums = {}      # metric_name -> weighted sum
    counts = {}    # metric_name -> weight sum

    def _accumulate(metrics_dict, n_agents_valid, n_points_valid, prefix=""):
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = float(v)
            name = prefix + k
            if ("macro" in k) or (k in {"minADE_K","minFDE_K","MR@m"}):
                w = max(n_agents_valid, 1)
            elif "micro" in k:
                w = max(n_points_valid, 1)
            else:
                # fallback: weight by #agents
                w = max(n_agents_valid, 1)
            sums[name] = sums.get(name, 0.0) + v * w
            counts[name] = counts.get(name, 0) + w

    with torch.inference_mode():
        for batch in dl:
            ag_h  = batch["agents_hist_xy"].to(device)        # (B,Na,Th,2)
            ag_hm = batch["agents_hist_mask"].to(device)      # (B,Na,Th)
            eg_h  = batch["ego_hist_xy"].to(device)           # (B,Th,2)
            eg_hm = batch["ego_hist_mask"].to(device)         # (B,Th)
            gt    = batch["agents_fut_xy"].to(device)         # (B,Na,Tf,2)
            gtm   = batch["agents_fut_mask"].to(device)       # (B,Na,Tf)

            # 有效 agent（至少一個未來點有效）
            n_agents_valid = int(gtm.any(dim=-1).sum().item())
            # 有效點數（micro 權重）
            n_points_valid = int(gtm.sum().item())

            if not multirollout:
                pred = model(ag_h, ag_hm, eg_h, eg_hm)        # (B,Na,Tf,2)
                m = ade_fde_single(pred, gt, gtm)             # dict: macro/micro ADE/FDE
                _accumulate(m, n_agents_valid, n_points_valid, prefix="")
            else:
                pred_k, _ = model(ag_h, ag_hm, eg_h, eg_hm)   # (B,Na,K,Tf,2)
                # 也可看第 1 條的單模態視角（可選）
                m1 = ade_fde_single(pred_k[:, :, 0], gt, gtm)
                _accumulate(m1, n_agents_valid, n_points_valid, prefix="single_")
                # 正式 multi-rollout 指標
                m2 = metrics_multirollout(pred_k, gt, gtm, miss_thresh=args.miss_thresh)
                _accumulate(m2, n_agents_valid, n_points_valid, prefix="")

    # finalize weighted averages
    out = {k: (sums[k] / max(counts[k], 1)) for k in sums.keys()}

    print(json.dumps(out, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
