import torch
EOS = 1e-10


def normalize_adj(adj, mode='sym'):
    assert len(adj.shape) in [2, 3]
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.abs().sum(dim=-1, keepdim=False)) + EOS)
        if len(adj.shape) == 3:
            return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.abs().sum(dim=-1, keepdim=False) + EOS)
        if len(adj.shape) == 3:
            return inv_degree[:, :, None] * adj
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")
