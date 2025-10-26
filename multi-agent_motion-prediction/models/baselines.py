# models/baselines.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import amp  # 新增

def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask: (..., T) -> lengths (...,)
    return mask.long().sum(dim=-1).clamp(min=0)

def _encode_gru(gru, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    seq:  (N, T, D=2)
    mask: (N, T) bool
    return: (N, H*dir) 最後一層雙向拼接的 hidden（對於 length=0 的樣本回全 0）
    """
    N, T, D = seq.shape
    device = seq.device
    num_dir = 2 if gru.bidirectional else 1
    H = gru.hidden_size

    lens = _lengths_from_mask(mask)              # (N,)
    h_last = seq.new_zeros((N, H * num_dir))     # 先準備輸出緩衝

    nonzero = (lens > 0) #(N, ) bool
    if nonzero.any():
        idx_nz = torch.nonzero(nonzero, as_tuple=False).squeeze(1)  # 有效樣本 index, (M, ), long
        seq_nz = seq.index_select(0, idx_nz) #(M, T, D)
        len_nz = lens.index_select(0, idx_nz) #(M, )

        # 依長度排序（pack 需要）
        len_sorted, idx_sort = torch.sort(len_nz.cpu(), descending=True) #(M, ) (M, )
        seq_sorted = seq_nz.index_select(0, idx_sort.to(device)) #(M, T, D)

        packed = pack_padded_sequence(
            seq_sorted, lengths=len_sorted, batch_first=True, enforce_sorted=True
        ) #()
        with amp.autocast(device_type='cuda', enabled=False):
            _, h = gru(packed)                      # h: (num_layers*num_dir, M, H)

        # 取「最後一層」的雙向 hidden，拼成 (N', H*dir)
        h_last_layer = h[-num_dir:]             # (num_dir, M, H)
        h_last_layer = h_last_layer.transpose(0, 1).reshape(-1, H * num_dir) #(M, H*num_dir)

        # 還原 pack 的排序
        inv_sort = torch.argsort(idx_sort).to(device)
        h_last_nz = h_last_layer.index_select(0, inv_sort) #(M, H*num_dir)

        # 填回對應位置；零長度的樣本保持 0
        h_last.index_copy_(0, idx_nz, h_last_nz) #(N, H*num_dir)

    return h_last

class GRUSingleRollout(nn.Module):
    def __init__(self, Th: int, Tf: int, d_model: int = 128, hidden: int = 128,
                 num_layers: int=2, dropout: float = 0.1):
        super().__init__()
        self.Th, self.Tf = Th, Tf
        H = hidden // 2  # 雙向 => 總 hidden = 2H

        self.ag_enc = nn.GRU(2, H, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=0.0 if num_layers==1 else 0.1)#兩層 + 跨層 dropout
        self.ego_enc = nn.GRU(2, H, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=0.0 if num_layers==1 else 0.1)#兩層 + 跨層 dropout

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
