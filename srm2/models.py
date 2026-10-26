"""Dense backbones over per-field embeddings emb [B, F, D] and dense features x [B, Nd]. All return logits [B].

dlrm   : DLRM (dot interaction; bottom 256-128-D, top 256-128-1) -- the configuration used throughout.
dlrmv2 : MLPerf DLRM-DCNv2: bottom 512-256-D; low-rank DCNv2 cross (3 layers, rank 512) over
         [bottom(x), emb]; top 1024-1024-512-256-1.
dcnv2  : DCN-V2 (stacked): x0 = [x, emb]; low-rank cross (3 layers, rank 512); deep 1024-1024-512; linear head.
deepfm : DeepFM: first-order + FM second-order + DNN 400-400-400. The first-order weight of each feature is
         packed as the last coordinate of its embedding row (Adam is element-wise, so this is equivalent to
         a separate first-order table updated by the same optimizer); FM uses the first D-1 coordinates.
"""
import torch
from torch import nn


def mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DLRM(nn.Module):
    def __init__(self, n_dense, n_fields, dim):
        super().__init__()
        self.bottom = mlp([n_dense, 256, 128, dim])
        self.top = mlp([dim + (n_fields + 1) * n_fields // 2, 256, 128, 1], last_act=False)
        ij = torch.tril_indices(n_fields + 1, n_fields + 1, offset=-1)
        self.register_buffer('flat', ij[0] * (n_fields + 1) + ij[1])

    def forward(self, x, emb):
        dense = self.bottom(x); v = torch.cat((dense[:, None, :], emb), 1)
        z = torch.bmm(v, v.transpose(1, 2)).flatten(1).index_select(1, self.flat)
        return self.top(torch.cat((dense, z), 1)).flatten()


class LowRankCross(nn.Module):
    """DCN-V2 cross layers: x_{l+1} = x0 * (V(U x_l) + b) + x_l (low-rank)."""

    def __init__(self, d, layers=3, rank=512):
        super().__init__()
        self.U = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(layers)])
        self.V = nn.ModuleList([nn.Linear(rank, d) for _ in range(layers)])

    def forward(self, x0):
        x = x0
        for U, V in zip(self.U, self.V):
            x = x0 * V(U(x)) + x
        return x


class DLRMv2(nn.Module):
    def __init__(self, n_dense, n_fields, dim):
        super().__init__()
        self.bottom = mlp([n_dense, 512, 256, dim])
        d = dim * (n_fields + 1)
        self.cross = LowRankCross(d, 3, 512)
        self.top = mlp([d, 1024, 1024, 512, 256, 1], last_act=False)

    def forward(self, x, emb):
        x0 = torch.cat((self.bottom(x), emb.flatten(1)), 1)
        return self.top(self.cross(x0)).flatten()


class DCNv2(nn.Module):
    def __init__(self, n_dense, n_fields, dim):
        super().__init__()
        d = n_dense + dim * n_fields
        self.cross = LowRankCross(d, 3, 512)
        self.deep = mlp([d, 1024, 1024, 512])
        self.head = nn.Linear(512, 1)

    def forward(self, x, emb):
        x0 = torch.cat((x, emb.flatten(1)), 1)
        return self.head(self.deep(self.cross(x0))).flatten()


class DeepFM(nn.Module):
    def __init__(self, n_dense, n_fields, dim):
        super().__init__()
        self.lin_dense = nn.Linear(n_dense, 1)
        self.dnn = mlp([n_dense + n_fields * dim, 400, 400, 400, 1], last_act=False)

    def forward(self, x, emb):
        first = emb[:, :, -1].sum(1) + self.lin_dense(x).flatten()
        v = emb[:, :, :-1]
        s = v.sum(1)
        fm = 0.5 * (s * s - (v * v).sum(1)).sum(1)
        deep = self.dnn(torch.cat((x, emb.flatten(1)), 1)).flatten()
        return first + fm + deep


class DLRMMT(DLRM):
    """Shared-bottom multi-task DLRM: same bottom / interaction / top MLP, one output logit per task. Returns [B, T]."""

    def __init__(self, n_dense, n_fields, dim, n_tasks=3):
        super().__init__(n_dense, n_fields, dim)
        self.top = mlp([dim + (n_fields + 1) * n_fields // 2, 256, 128, n_tasks], last_act=False)

    def forward(self, x, emb):
        dense = self.bottom(x); v = torch.cat((dense[:, None, :], emb), 1)
        z = torch.bmm(v, v.transpose(1, 2)).flatten(1).index_select(1, self.flat)
        return self.top(torch.cat((dense, z), 1))


class MMoE(nn.Module):
    """MMoE (Ma et al., KDD'18) over DLRM features: input = [bottom(x), pairwise dot interactions]; E experts
    (MLP 256-128), one softmax gate per task, task towers 64-1. Returns [B, T]."""

    def __init__(self, n_dense, n_fields, dim, n_tasks=3, n_experts=4):
        super().__init__()
        self.T, self.E = n_tasks, n_experts
        self.bottom = mlp([n_dense, 256, 128, dim])
        ij = torch.tril_indices(n_fields + 1, n_fields + 1, offset=-1)
        self.register_buffer('flat', ij[0] * (n_fields + 1) + ij[1])
        d = dim + (n_fields + 1) * n_fields // 2
        self.experts = nn.ModuleList([mlp([d, 256, 128]) for _ in range(n_experts)])
        self.gates = nn.Linear(d, n_tasks * n_experts)
        self.towers = nn.ModuleList([mlp([128, 64, 1], last_act=False) for _ in range(n_tasks)])

    def forward(self, x, emb):
        dense = self.bottom(x); v = torch.cat((dense[:, None, :], emb), 1)
        z = torch.bmm(v, v.transpose(1, 2)).flatten(1).index_select(1, self.flat)
        h = torch.cat((dense, z), 1)
        ex = torch.stack([e(h) for e in self.experts], 1)                                   # [B, E, 128]
        g = torch.softmax(self.gates(h).view(-1, self.T, self.E), -1)                       # [B, T, E]
        mix = torch.bmm(g, ex)                                                              # [B, T, 128]
        return torch.cat([tw(mix[:, t]) for t, tw in enumerate(self.towers)], 1)           # [B, T]


MODELS = {'dlrm': DLRM, 'dlrmv2': DLRMv2, 'dcnv2': DCNv2, 'deepfm': DeepFM, 'dlrm_mt': DLRMMT, 'mmoe': MMoE}
MULTI_TASK = ('dlrm_mt', 'mmoe')
