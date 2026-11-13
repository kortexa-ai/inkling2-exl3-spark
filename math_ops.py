"""Pinned Inkling arithmetic used by the two-stage text adapter."""
import torch
import torch.nn.functional as F

def linear(x, w, *, output_dtype=torch.bfloat16):
    if x.is_cuda:
        y = torch.mm(x.bfloat16(), w.bfloat16().T, out_dtype=torch.float32)
    else:
        y = x.float() @ w.float().T
    return y.to(output_dtype)

def rmsnorm(x, weight=None, eps=1e-6, dtype=torch.bfloat16):
    y = x.float()
    squares = y.square().sum(-1, keepdim=True)
    y = y * torch.rsqrt(squares / torch.full_like(squares, y.shape[-1]) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(dtype)

def l2clip(x, scale=None, threshold=512**0.5):
    y = x.float()
    if scale is not None:
        y = y * scale.float()
    coefficient = (threshold / (y.square().sum(-1, keepdim=True).sqrt() + 1e-6)).clamp(max=1)
    return (y * coefficient).to(x.dtype)

def attention(q, k, v, rel, qpos, kpos, window=1023):
    """Causal local grouped-query attention, including compact relative bias."""
    group = q.shape[1] // k.shape[1]
    keys = k.float().repeat_interleave(group, 1)
    values = v.float().repeat_interleave(group, 1)
    scores = torch.einsum("thd,shd->hts", q.float(), keys) * q.shape[-1]**-0.5
    distance = qpos[:, None] - kpos[None, :]
    inside = (distance >= 0) & (distance < rel.shape[-1])
    bias = rel.float().permute(1, 0, 2)[:, torch.arange(len(q), device=q.device)[:, None],
                                                    distance.clamp(0, rel.shape[-1] - 1)]
    scores = scores + torch.where(inside[None], bias, 0)
    mask = distance < 0
    if window is not None:
        mask = mask | (distance > window)
    scores.masked_fill_(mask[None], -torch.inf)
    lse = scores.logsumexp(-1).T
    output = torch.einsum("hts,shd->thd", scores.softmax(-1), values)
    return output, lse

