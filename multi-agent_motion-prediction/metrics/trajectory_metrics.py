# metrics/trajectory_metrics.py
from __future__ import annotations
import torch
from typing import Dict, Tuple

def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return numer / (denom + eps)

def _last_valid_index(mask_bnT: torch.Tensor) -> torch.Tensor:
    """
    mask_bnT: (B,Na,T) bool. 回傳 (B,Na) 的最後 True 的 index；若全 False -> -1。
    """
    B, Na, T = mask_bnT.shape
    idx = torch.arange(T, device=mask_bnT.device).view(1, 1, T).expand(B, Na, T)
    # 將非有效位置設為 -1，取最大
    scores = torch.where(mask_bnT, idx, torch.full_like(idx, -1))
    return scores.max(dim=-1).values  # (B,Na), -1 表示沒有有效步

def ade_fde_single(
    pred: torch.Tensor,  # (B,Na,Tf,2)
    gt: torch.Tensor,    # (B,Na,Tf,2)
    mask: torch.Tensor   # (B,Na,Tf) bool
) -> Dict[str, torch.Tensor]:
    """
    回傳：
      - macro_ADE / micro_ADE
      - macro_FDE / micro_FDE（對 FDE 其實 macro/micro 相同，皆對「有末步」的 agent 平均）
      - per_agent_ADE/FDE（(B,Na)；供日後分析）
    """
    assert pred.shape == gt.shape
    assert mask.shape == pred.shape[:3]
    B, Na, Tf, _ = pred.shape

    d = torch.linalg.norm(pred - gt, dim=-1)  # (B,Na,Tf)

    # ADE
    valid_counts = mask.sum(dim=-1).clamp(min=1)            # (B,Na)
    ade_per_agent = _safe_div((d * mask).sum(-1), valid_counts)  # (B,Na)

    has_any = mask.any(dim=-1)  # (B,Na)
    macro_ADE = ade_per_agent[has_any].mean() if has_any.any() else torch.tensor(0.0, device=pred.device)
    micro_ADE = _safe_div((d * mask).sum(), mask.sum()) if mask.any() else torch.tensor(0.0, device=pred.device)

    # FDE：取每個 agent 的最後有效一步
    last_idx = _last_valid_index(mask)  # (B,Na), -1 代表沒有
    has_last = last_idx >= 0
    # gather last distances
    gather_idx = last_idx.clamp(min=0).unsqueeze(-1)  # (B,Na,1)
    d_last = d.gather(dim=2, index=gather_idx).squeeze(-1)  # (B,Na)
    d_last = torch.where(has_last, d_last, torch.zeros_like(d_last))
    # macro/micro（對 FDE 等價，因只對有末步的 agent 平均）
    macro_FDE = d_last[has_last].mean() if has_last.any() else torch.tensor(0.0, device=pred.device)
    micro_FDE = macro_FDE

    return {
        "macro_ADE": macro_ADE,
        "micro_ADE": micro_ADE,
        "macro_FDE": macro_FDE,
        "micro_FDE": micro_FDE,
        "per_agent_ADE": ade_per_agent,  # (B,Na)
        "per_agent_FDE": d_last,         # (B,Na)（無末步者為 0）
        "per_agent_has": has_any,        # (B,Na)
    }

def metrics_multirollout(
    pred_k: torch.Tensor,    # (B,Na,K,Tf,2)
    gt: torch.Tensor,        # (B,Na,Tf,2)
    mask: torch.Tensor,      # (B,Na,Tf) bool
    miss_thresh: float = 2.0 # MR@m 的 m（公尺）
) -> Dict[str, torch.Tensor]:
    """
    multirollout 指標（僅在你有 K 條輸出時使用）：
      - minADE_K / minFDE_K：對每個 agent 在 K 條中取最小，再做 macro 平均
      - MR@m：minFDE_K > m 的比例
    """
    B, Na, K, Tf, _ = pred_k.shape

    # L2 over time
    d = torch.linalg.norm(pred_k - gt.unsqueeze(2), dim=-1)  # (B,Na,K,Tf)
    # ADE per (B,Na,K)
    valid_counts = mask.sum(dim=-1).clamp(min=1).unsqueeze(2)  # (B,Na,1)
    ade_bnk = (d * mask.unsqueeze(2)).sum(-1) / valid_counts   # (B,Na,K)

    # FDE per (B,Na,K)
    last_idx = _last_valid_index(mask)  # (B,Na)
    has_last = last_idx >= 0
    gather_idx = last_idx.clamp(min=0).view(B, Na, 1, 1).expand(B, Na, K, 1)
    fde_bnk = d.gather(dim=3, index=gather_idx).squeeze(-1)  # (B,Na,K)

    # min over K
    minADE_per_agent, _ = ade_bnk.min(dim=2)   # (B,Na)
    minFDE_per_agent, _ = fde_bnk.min(dim=2)   # (B,Na)

    has_any = mask.any(dim=-1)
    minADE_K = minADE_per_agent[has_any].mean() if has_any.any() else torch.tensor(0.0, device=pred_k.device)
    minFDE_K = minFDE_per_agent[has_last].mean() if has_last.any() else torch.tensor(0.0, device=pred_k.device)

    # MR@m
    misses = (minFDE_per_agent > miss_thresh) & has_last
    MR_m = _safe_div(misses.sum(), has_last.sum()) if has_last.any() else torch.tensor(0.0, device=pred_k.device)

    return {
        "minADE_K": minADE_K,
        "minFDE_K": minFDE_K,
        "MR@m": MR_m,
        "per_agent_minADE": minADE_per_agent,
        "per_agent_minFDE": minFDE_per_agent,
    }
