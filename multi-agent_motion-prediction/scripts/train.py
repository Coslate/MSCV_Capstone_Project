# scripts/train.py
import os, argparse, math, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.packed_nuscenes_dataset import PackedNuScenesDataset, packed_nuscenes_collate
from models.baselines import GRUSingleRollout, GRUMultiRollout
from metrics.trajectory_metrics import ade_fde_single, metrics_multirollout
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch import amp
from tqdm import tqdm

# ----------------- Loss helpers -----------------
def masked_l2_loss(pred, gt, mask):
    """
    pred, gt: (B,Na,Tf,2)
    mask:     (B,Na,Tf) bool
    """
    # Expand mask to coords and zero-out invalid entries BEFORE differencing
    valid = mask.unsqueeze(-1)                      # (B,Na,Tf,1)
    pred_v = torch.where(valid, pred, torch.zeros_like(pred))
    gt_v   = torch.where(valid, gt,   torch.zeros_like(gt))   # <-- kills NaNs at invalid spots

    diff = pred_v - gt_v                                # (B,Na,Tf,2)
    l2   = torch.linalg.norm(diff, dim=-1)              # (B,Na,Tf)

    denom = mask.sum().clamp(min=1)
    return l2.sum() / denom

def best_of_k_loss(pred_k, gt, mask, logits=None, lambda_cls: float = 0.1):
    """
    pred_k: (B,Na,K,Tf,2)   gt: (B,Na,Tf,2)    mask: (B,Na,Tf)
    Winner-take-all best-of-K regression + optional class CE on argmin.
    """
    B, Na, K, Tf, _ = pred_k.shape

    valid = mask.unsqueeze(2).unsqueeze(-1)             # (B,Na,K,Tf,1)
    gt_e  = gt.unsqueeze(2)                             # (B,Na,1,Tf,2)
    pred_v = torch.where(valid, pred_k, torch.zeros_like(pred_k))
    gt_v   = torch.where(valid, gt_e,   torch.zeros_like(gt_e))

    diff   = pred_v - gt_v                              # (B,Na,K,Tf,2)
    d      = torch.linalg.norm(diff, dim=-1)            # (B,Na,K,Tf)

    counts = mask.sum(dim=-1).clamp(min=1).unsqueeze(2) # (B,Na,1)
    ade_bk = d.sum(dim=-1) / counts                     # (B,Na,K)

    min_val, min_idx = ade_bk.min(dim=2)                # (B,Na)
    reg = min_val.mean()

    if logits is None:
        return reg, torch.tensor(0.0, device=reg.device), min_idx

    ce = nn.CrossEntropyLoss()
    cls = ce(logits.view(-1, K), min_idx.view(-1))
    return reg + lambda_cls * cls, cls, min_idx

# ----------------- Evaluation -----------------
def run_val(model, dl, device, multirollout: bool,
            miss_thresh: float = 2.0, K: int = 6,
            wandb_run=None, step=None, show_pbar: bool=True, desc: str="Val",
            lambda_cls: float = 0.1):
    model.eval()
    meter = {}
    loss_total = 0.0
    cls_total  = 0.0
    n_batches  = 0

    iterator = tqdm(dl, desc=desc, dynamic_ncols=True, leave=False) if show_pbar else dl
    with torch.no_grad():
        for batch in iterator:
            ag_h = batch["agents_hist_xy"].to(device)
            ag_hm= batch["agents_hist_mask"].to(device)
            eg_h = batch["ego_hist_xy"].to(device)
            eg_hm= batch["ego_hist_mask"].to(device)
            gt   = batch["agents_fut_xy"].to(device)
            gtm  = batch["agents_fut_mask"].to(device)

            if not multirollout:
                # forward
                pred = model(ag_h, ag_hm, eg_h, eg_hm)  # (B,Na,Tf,2)
                # val loss (same as train objective for single rollout)
                loss_b = masked_l2_loss(pred, gt, gtm)

                # metrics
                m = ade_fde_single(pred, gt, gtm)

                # accumulate
                loss_total += float(loss_b.item())
                n_batches  += 1
                for k, v in m.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        meter[k] = meter.get(k, 0.0) + float(v)
            else:
                # forward (need logits for best_of_k val loss)
                pred_k, logits = model(ag_h, ag_hm, eg_h, eg_hm)  # (B,Na,K,Tf,2), (B,Na,K)

                # val loss (WTA + optional cls)
                loss_b, cls_b, _ = best_of_k_loss(pred_k, gt, gtm, logits, lambda_cls=lambda_cls)

                # metrics
                m1 = ade_fde_single(pred_k[:, :, 0], gt, gtm)  # single-view (K=0) metrics
                for k, v in m1.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        meter["single_" + k] = meter.get("single_" + k, 0.0) + float(v)

                m2 = metrics_multirollout(pred_k, gt, gtm, miss_thresh=miss_thresh)
                for k, v in m2.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        meter[k] = meter.get(k, 0.0) + float(v)

                # accumulate
                loss_total += float(loss_b.item())
                cls_total  += float(cls_b.item())
                n_batches  += 1

    # average over batches
    if n_batches > 0:
        meter["loss"] = loss_total / n_batches
        if multirollout:
            meter["cls_loss"] = cls_total / n_batches
        for k in list(meter.keys()):
            if k not in ("loss", "cls_loss"):
                meter[k] /= n_batches

    # log to W&B if provided
    if wandb_run is not None:
        wandb_run.log({f"val/{k}": v for k, v in meter.items()}, step=step)

    model.train()
    return meter

# ----------------- CKPT helpers -----------------
def _resolve_ckpt_path(p: str, prefer: str = "last.pt"):
    """允許傳入目錄或檔案：若是目錄，預設找 prefer（last.pt 或 best.pt）"""
    if p is None:
        return None
    if os.path.isdir(p):
        cand = os.path.join(p, prefer)
        return cand if os.path.isfile(cand) else None
    return p if os.path.isfile(p) else None

def _count_params(model):
    return sum(p.numel() for p in model.parameters())

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest",   required=True)
    ap.add_argument("--test_manifest",  default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=500, help="validate every N iterations")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--multirollout", action="store_true")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lambda_cls", type=float, default=0.1)
    ap.add_argument("--miss_thresh", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp_dtype", type=str, default="fp16",
                choices=["fp16", "bf16", "off"],
                help="AMP precision on CUDA: fp16 (default), bf16, or off")

    ap.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="warmup steps as a fraction of total training steps")
    ap.add_argument("--warmup_steps", type=int, default=None,
                    help="override warmup_ratio with an explicit number of steps")
    ap.add_argument("--min_lr", type=float, default=1e-6,
                    help="cosine annealing floor")
    ap.add_argument("--no_lr_sched", action="store_true",
                    help="disable LR scheduler (keep constant LR)")
    ap.add_argument("--warmup_init_lr", type=float, default=None,
                    help="LR to start warmup from (default: lr * 1e-2).")

    # checkpoint I/O
    ap.add_argument("--out_dir", type=str, default="checkpoints",
                    help="directory to store checkpoints (best.pt / last.pt)")
    ap.add_argument("--resume", type=str, default=None,
                    help="path or directory to resume from (loads model+optim+scaler+it/epoch/best)")
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="path or directory to initialize model weights only (no optimizer/scaler)")
    ap.add_argument("--save_every", type=int, default=0,
                    help="save last.pt every N iterations (0=only on validation/end)")

    # ----- Weights & Biases -----
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="traj-pred")
    ap.add_argument("--wandb_entity",  type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group",    type=str, default=None)
    ap.add_argument("--wandb_tags",     nargs="*", default=None)
    ap.add_argument("--wandb_id",       type=str, default=None, help="resume to this run id")
    ap.add_argument("--log_every",      type=int, default=50, help="log train stats every N iters")
    ap.add_argument("--watch_model",    action="store_true", help="wandb.watch(model) gradients/params")
    ap.add_argument("--wandb_ckpt_artifacts", action="store_true", help="upload checkpoints as W&B artifacts")

    args = ap.parse_args()

    # 選 AMP dtype
    if args.amp_dtype == "off" or torch.device(args.device).type != "cuda":
        amp_enabled = False
        amp_dtype = None
    else:
        amp_enabled = True
        amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    # GradScaler：fp16 才需要；bf16 通常不需要
    use_scaler = (amp_enabled and amp_dtype is torch.float16)
    scaler = amp.GradScaler('cuda', enabled=use_scaler)

    #（可選）開 TF32 提升吞吐（不影響 AMP dtype）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[AMP] enabled={amp_enabled} dtype={amp_dtype} scaler={use_scaler}")    
    print("[AMP] current GPU autocast dtype =", torch.get_autocast_gpu_dtype())

    # Datasets / Loaders (train/val/test splits on manifest.jsonl)
    tr_ds = PackedNuScenesDataset(args.train_manifest)
    va_ds = PackedNuScenesDataset(args.val_manifest)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=packed_nuscenes_collate, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=packed_nuscenes_collate, pin_memory=True)

    Th = tr_ds.Th; Tf = tr_ds.Tf
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.multirollout:
        model = GRUSingleRollout(Th=Th, Tf=Tf, hidden=args.hidden,
                                 num_layers=args.num_layers, d_model=args.d_model).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        model = GRUMultiRollout(Th=Th, Tf=Tf, K=args.K, hidden=args.hidden,
                                num_layers=args.num_layers, d_model=args.d_model).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- LR schedule: linear warmup -> cosine anneal ----
    steps_per_epoch = max(1, len(tr_dl))
    total_steps = max(args.epochs * steps_per_epoch, 1)

    if args.warmup_steps is not None:
        warmup_steps = int(args.warmup_steps)
    else:
        warmup_steps = int(total_steps * max(args.warmup_ratio, 0.0))

    warmup_steps = max(min(warmup_steps, total_steps - 1), 1)  # clamp
    scheduler = None
    if not args.no_lr_sched:
        if warmup_steps > 0:
            # choose warmup start LR (default 1% of base LR)
            warmup_init_lr = args.warmup_init_lr if args.warmup_init_lr is not None else args.lr * 1e-2
            start_factor = float(warmup_init_lr / max(args.lr, 1e-12))  # must be (0,1]
            start_factor = min(max(start_factor, 1e-8), 1.0)

            sched_warm = LinearLR(optim, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
            sched_cos  = CosineAnnealingLR(optim, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr)
            scheduler  = SequentialLR(optim, schedulers=[sched_warm, sched_cos], milestones=[warmup_steps])
        else:
            scheduler  = CosineAnnealingLR(optim, T_max=total_steps, eta_min=args.min_lr)

    print(f"[LR] steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ★ init / resume
    start_it = 0
    start_epoch = 0
    best_val = float("inf")

    # only loading model weights (for fine-tuning)
    init_path = _resolve_ckpt_path(args.init_ckpt, prefer="best.pt")
    if init_path:
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[init] loaded model weights from {init_path}")

    # resume training (loading the training status, optimization status to resume)
    resume_path = _resolve_ckpt_path(args.resume, prefer="last.pt")
    maybe_wandb_id_from_ckpt = None
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optim" in ckpt: optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_it = int(ckpt.get("it", 0))
        start_epoch = int(ckpt.get("epoch", 0))

        # Try to restore scheduler state; if absent, fast-forward by start_it
        if scheduler is not None:
            if resume_path and "scheduler" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    print("[resume] scheduler state restored.")
                except Exception as e:
                    print(f"[resume] scheduler load failed ({e}); fast-forward {start_it} steps.")
                    for _ in range(start_it):
                        scheduler.step()
            elif start_it > 0:
                print(f"[resume] no scheduler state; fast-forward {start_it} steps.")
                for _ in range(start_it):
                    scheduler.step()

        best_val = float(ckpt.get("best_score", float("inf")))
        maybe_wandb_id_from_ckpt = ckpt.get("wandb_id", None)
        print(f"[resume] from {resume_path} | it={start_it} epoch={start_epoch} best={best_val:.4f}")

    # ----- W&B init -----
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise RuntimeError("Please `pip install wandb` to use --wandb") from e

        # 選 run id：優先 args.wandb_id；否則從 ckpt 繼承；否則 None（新 run）
        run_id = args.wandb_id or maybe_wandb_id_from_ckpt
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            id=run_id,
            resume="allow" if run_id else None,
            config={
                **vars(args),
                "Th": Th, "Tf": Tf,
                "params": _count_params(model),
                "resume_path": resume_path,
                "init_path": init_path
            }
        )
        if args.watch_model:
            wandb.watch(model, log="gradients", log_freq=args.log_every)

    it = start_it

    def _save(path):
        torch.save({
            "it": it, "epoch": ep, "model": model.state_dict(),
            "optim": optim.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "args": vars(args), "best_score": best_val,
            "wandb_id": (wandb_run.id if wandb_run is not None else None)
        }, path)
        # 上傳 artifact（可選）
        if wandb_run is not None and args.wandb_ckpt_artifacts:
            import wandb as _wandb
            art = _wandb.Artifact(name=f"ckpt-{_wandb.util.generate_id()}",
                                  type="model",
                                  metadata={"step": it, "epoch": ep, "best_score": best_val})
            art.add_file(path, name=os.path.basename(path))
            wandb_run.log_artifact(art)

    if wandb_run is not None:
        wandb_run.log({"amp/dtype": str(torch.get_autocast_gpu_dtype())}, step=it)

    # ----------------- Train Loop -----------------
    for ep in range(start_epoch, args.epochs):
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            it += 1
            ag_h = batch["agents_hist_xy"].to(device)       # (B,Na,Th,2)
            ag_hm= batch["agents_hist_mask"].to(device)     # (B,Na,Th)
            eg_h = batch["ego_hist_xy"].to(device)          # (B,Th,2)
            eg_hm= batch["ego_hist_mask"].to(device)        # (B,Th)
            gt   = batch["agents_fut_xy"].to(device)        # (B,Na,Tf,2)
            gtm  = batch["agents_fut_mask"].to(device)      # (B,Na,Tf)

            optim.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                if not args.multirollout:
                    pred = model(ag_h, ag_hm, eg_h, eg_hm)   # (B,Na,Tf,2)
                    loss = masked_l2_loss(pred, gt, gtm)
                    cls = None
                else:
                    pred_k, logits = model(ag_h, ag_hm, eg_h, eg_hm)  # (B,Na,K,Tf,2), (B,Na,K)
                    loss, cls, _ = best_of_k_loss(pred_k, gt, gtm, logits, lambda_cls=args.lambda_cls)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            # 更新進度列的 postfix
            if it % 10 == 0:
                lr = optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", it=it)

            # wandb log (train)
            if wandb_run is not None and (it % args.log_every == 0):
                lr = optim.param_groups[0]["lr"]
                log_dict = {"train/loss": loss.item(), "lr": lr, "it": it, "epoch": ep}
                if cls is not None:
                    log_dict["train/cls_loss"] = float(cls.item())
                wandb_run.log(log_dict, step=it)

            # frequently saving last.pt
            if args.save_every > 0 and it % args.save_every == 0:
                _save(os.path.join(args.out_dir, "last.pt"))

            # validation
            if it % args.val_every == 0:
                val_m = run_val(model, va_dl, device, args.multirollout, args.miss_thresh, args.K,
                                wandb_run=wandb_run, step=it, show_pbar=True, desc=f"Val@it {it}",
                                lambda_cls=args.lambda_cls)
                key = ("minADE_K" if args.multirollout else "macro_ADE")
                score = val_m.get(key, float("inf"))
                print(f"[VAL it {it}] {json.dumps(val_m, indent=2)}")

                # save last.pt every val
                _save(os.path.join(args.out_dir, "last.pt"))

                # save the best
                if score < best_val:
                    best_val = score
                    _save(os.path.join(args.out_dir, "best.pt"))
                    print(f"** saved best checkpoint to {args.out_dir}/best.pt (score={best_val:.4f})")

        # epoch 結束也存一下 last.pt
        _save(os.path.join(args.out_dir, "last.pt"))

    # optional: 最後再驗一次
    val_m = run_val(model, va_dl, device, args.multirollout, args.miss_thresh, args.K,
                    wandb_run=wandb_run, step=it, lambda_cls=args.lambda_cls)
    print("[VAL-final]", val_m)

    # 若有 test_manifest，就載最好模型去跑
    if args.test_manifest:
        ts_ds = PackedNuScenesDataset(args.test_manifest)
        ts_dl = DataLoader(ts_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=packed_nuscenes_collate, pin_memory=True)
        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_route=device if False else device)
        model.load_state_dict(ckpt["model"])
        test_m = run_val(model, ts_dl, device, args.multirollout, args.miss_thresh, args.K,
                         wandb_run=wandb_run, step=it)
        print("[TEST]", test_m)

    # finish wandb
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
