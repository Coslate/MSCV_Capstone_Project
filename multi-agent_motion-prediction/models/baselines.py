# models/baselines.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask: (..., T) -> lengths (...,)
    return mask.long().sum(dim=-1).clamp(min=0)

def _encode_gru(gru: nn.GRU, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    seq: (N,T,2), mask: (N,T) bool
    回傳: (N, H*2) 的雙向「最後一層」隱向量（長度=0 的樣本回 0 向量）
    支援任意 num_layers；h 形狀為 (num_layers*2, N, H)，h[-2:] 就是最後一層的 fwd/back。
    """

    N, T, D = seq.shape
    lens = _lengths_from_mask(mask)  # (N,)
    # 避免空序列：全 0 也能 pack，但需排序
    lens_cpu = lens.cpu()
    lens_sorted, idx_sort = torch.sort(lens_cpu, descending=True)
    idx_unsort = torch.argsort(idx_sort).to(seq.device)
    seq_sorted = seq.index_select(0, idx_sort.to(seq.device))
    # pack
    packed = pack_padded_sequence(seq_sorted, lengths=lens_sorted, batch_first=True, enforce_sorted=True)
    _, h = gru(packed)  # h: (num_layers*2, N, H)
    h_last = h[-2:].transpose(0,1).reshape(N, -1)  # 取最後一層雙向 -> (N, 2H)
    # 還原順序
    h_last = h_last.index_select(0, idx_unsort)
    # 對於 length=0 的樣本，置零（pack 裡會當成 0）
    zero_mask = (lens == 0).unsqueeze(-1) #(N, 1)
    h_last = torch.where(zero_mask, torch.zeros_like(h_last), h_last) #(N, H*2)
    return h_last

class GRUSingleRollout(nn.Module):
    def __init__(self, Th: int, Tf: int, d_model: int = 128, hidden: int = 128,
                 num_layers: int=2, dropout: float = 0.1):
        super().__init__()
        self.Th, self.Tf = Th, Tf
        H = hidden // 2  # 雙向 => 總 hidden = 2H

        self.ag_enc = nn.GRU(
            input_size=2, hidden_size=H,
            batch_first=True, bidirectional=True,
            num_layers=num_layers, dropout=dropout       #兩層 + 跨層 dropout
        )
        self.ego_enc = nn.GRU(
            input_size=2, hidden_size=H,
            batch_first=True, bidirectional=True,
            num_layers=num_layers, dropout=dropout
        )

        self.fuse = nn.Sequential(
            nn.Linear(4*H, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(d_model, Tf*2)

    def forward(self,
                agents_hist_xy: torch.Tensor,    # (B,Na,Th,2)
                agents_hist_mask: torch.Tensor,  # (B,Na,Th)
                ego_hist_xy: torch.Tensor,       # (B,Th,2)
                ego_hist_mask: torch.Tensor      # (B,Th)
                ) -> torch.Tensor:
        B, Na, Th, _ = agents_hist_xy.shape
        Tf = self.Tf

        # encode agents（展平為 N=B*Na）
        x = agents_hist_xy.view(B*Na, Th, 2)
        m = agents_hist_mask.view(B*Na, Th)
        h_ag = _encode_gru(self.ag_enc, x, m)  # (B*Na, 2H)

        # encode ego（重複 broadcast 給每個 agent）
        h_ego = _encode_gru(self.ego_enc, ego_hist_xy, ego_hist_mask)  # (B, 2H)
        h_ego = h_ego.unsqueeze(1).expand(B, Na, -1).contiguous().view(B*Na, -1)  # (B*Na, 2H)

        z = torch.cat([h_ag, h_ego], dim=-1)   # (B*Na, 4H)
        z = self.fuse(z)                       # (B*Na, d_model)
        out = self.head(z).view(B, Na, Tf, 2)  # (B,Na,Tf,2)
        return out

class GRUMultiRollout(nn.Module):
    def __init__(self, Th: int, Tf: int, K: int = 6, d_model: int = 128, hidden: int = 128,
                  num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.Th, self.Tf, self.K = Th, Tf, K
        H = hidden // 2

        self.ag_enc = nn.GRU(
            input_size=2, hidden_size=H,
            batch_first=True, bidirectional=True,
            num_layers=num_layers, dropout=dropout       # 兩層
        )
        self.ego_enc = nn.GRU(
            input_size=2, hidden_size=H,
            batch_first=True, bidirectional=True,
            num_layers=num_layers, dropout=dropout       # 兩層
        )

        self.fuse = nn.Sequential(
            nn.Linear(4*H, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.traj_head = nn.Linear(d_model, K*Tf*2)  # K 條
        self.logit_head= nn.Linear(d_model, K)       # 信心 logits

    def forward(self,
                agents_hist_xy: torch.Tensor,    # (B,Na,Th,2)
                agents_hist_mask: torch.Tensor,  # (B,Na,Th)
                ego_hist_xy: torch.Tensor,       # (B,Th,2)
                ego_hist_mask: torch.Tensor      # (B,Th)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Na, Th, _ = agents_hist_xy.shape
        Tf, K = self.Tf, self.K

        x = agents_hist_xy.view(B*Na, Th, 2)
        m = agents_hist_mask.view(B*Na, Th)
        h_ag = _encode_gru(self.ag_enc, x, m)

        h_ego = _encode_gru(self.ego_enc, ego_hist_xy, ego_hist_mask)  # (B,2H)
        h_ego = h_ego.unsqueeze(1).expand(B, Na, -1).contiguous().view(B*Na, -1)

        z = torch.cat([h_ag, h_ego], dim=-1)
        z = self.fuse(z)

        traj = self.traj_head(z).view(B, Na, K, Tf, 2)  # (B,Na,K,Tf,2)
        logits = self.logit_head(z).view(B, Na, K)      # (B,Na,K)
        return traj, logits
