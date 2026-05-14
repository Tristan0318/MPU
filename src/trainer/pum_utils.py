import hashlib
import json
import math
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional



# Utility Function
def construct_seed64(master: int, *keys) -> int:
    """
    Construct a 64-bit integer seed from (master, *keys), ensuring deterministic
    and consistent behavior across processes/devices.
    """
    payload = json.dumps([int(master)] + list(keys), separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 64) - 1)


#cpu for stable
@torch.no_grad()
def compute_layerwise_sigma(
    sd_ref: "OrderedDict[str, torch.Tensor]",
    layer_groups: "Dict[str, List[str]]",
    kappa: float,
    base_sd: "OrderedDict[str, torch.Tensor]" = None,
) -> "Dict[str, float]":
    """
    Compute σ_l on CPU for stability and to avoid GPU memory overhead.

    For each block l:
      v_l = sd_ref - base_sd; if base_sd is None, v_l = sd_ref
      σ_l = kappa * RMS(v_l)
    """
    sigma = {}
    for lk, names in layer_groups.items():
        sumsq = 0.0
        count = 0
        for n in names:
            if n not in sd_ref:
                continue
            t = sd_ref[n]
            if not torch.is_tensor(t):
                continue

            # --- force CPU/FP32 statistics ---
            t_cpu = t.detach().to("cpu", dtype=torch.float32)

            if base_sd is None:
                v_cpu = t_cpu
            else:
                b = base_sd.get(n, None)
                if b is None or (not torch.is_tensor(b)):
                    # if missing, skip to be safe (or treat as zeros if you prefer)
                    continue
                b_cpu = b.detach().to("cpu", dtype=torch.float32)
                v_cpu = t_cpu - b_cpu

            sumsq += (v_cpu * v_cpu).sum().item()
            count += v_cpu.numel()

        sigma[lk] = 0.0 if count == 0 else float(kappa) * (sumsq / count) ** 0.5
    return sigma


# Per-layer zero-sum noise generation with α-scaling
@torch.no_grad()
def generate_zero_sum_noises(
    sd_ref: OrderedDict,
    layer_groups: dict,
    sigmas: dict,
    m: int,
    alphas,
    seed_round: int,
    device: torch.device,
):
    """
    Generate m structured correlated Gaussian noises with a zero-sum basis:
      eps0_{k} = sqrt(m/(m-1)) * (z_k - mean(z))
      eps_k    = alpha_k * eps0_k

    Robustness:
    - Sample z on CPU float32 (safe for all environments), then cast to parameter dtype on target device.
    - Seed per (round, layer_group, param_name, copy_id) to make determinism independent of key order.
    """
    assert m >= 2, "PUM requires m >= 2"
    assert len(alphas) == m, f"Expected {m} alphas, got {len(alphas)}"

    # Allocate output noises on the target device, matching each parameter shape/dtype.
    noises = []
    for _ in range(m):
        nd = OrderedDict()
        for name, t in sd_ref.items():
            if torch.is_tensor(t):
                nd[name] = torch.zeros_like(t, device=device, dtype=t.dtype)
        noises.append(nd)

    scale = math.sqrt(m / (m - 1))

    for lk, names in layer_groups.items():
        sigma = float(sigmas.get(lk, 0.0))
        if sigma == 0.0:
            continue

        for name in names:
            t = sd_ref.get(name, None)
            if not torch.is_tensor(t):
                continue

            # 1) sample z_k on CPU float32
            zs = []
            for k in range(m):
                g = torch.Generator(device="cpu")
                g.manual_seed(construct_seed64(seed_round, "noise", lk, name, k))
                z = torch.randn(t.shape, generator=g, device="cpu", dtype=torch.float32) * sigma
                zs.append(z)
            Z = torch.stack(zs, dim=0)               # [m, ...]
            mean = Z.mean(dim=0, keepdim=False)
            eps0 = (Z - mean) * scale                # still CPU fp32

            # 2) apply alpha and move to target device/dtype
            for k in range(m):
                noises[k][name].add_(eps0[k].to(device=device, dtype=t.dtype) * float(alphas[k]))

    return noises


# Reparameterization: orthogonal / rotation blocks
def generate_rand_orth_matrix(d: int, gen: torch.Generator, device, dtype):
    M = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    s = torch.sign(torch.diag(R))
    s = torch.where(s == 0, torch.ones_like(s), s)
    Q = Q @ torch.diag_embed(s)
    return Q.to(device=device, dtype=dtype)


def construct_rotation_matrix(theta: torch.Tensor):  # [...]->[...,2,2]
    theta32 = theta.to(torch.float32)
    c, s = torch.cos(theta32), torch.sin(theta32)
    return torch.stack(
        [torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2
    )  # [...,2,2]


# Sample T_{k,r} (attention components U and S_KV; without head permutation)
# pum_utils.py (NEW)


@torch.no_grad()
def sample_T_attention(
    H_Q: int,
    H_KV: int,
    d_h: int,
    rope_aware: bool,
    seed_round: int,
    k_copy: int,  # kept for API compatibility (not required)
    device="cpu",
    dtype=torch.float32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      idx_q: Long tensor [H_Q], mapping each Q head to a KV head index (π assignment)
      S_kv:  Tensor [H_KV, d_h, d_h] if rope_aware=False, else None
      cos:   Tensor [H_KV, d_h//2]   if rope_aware=True, else None
      sin:   Tensor [H_KV, d_h//2]   if rope_aware=True, else None

    NOTE: For HF Llama RoPE (rotate_half), the RoPE planes are (i, i + d_h/2).
          So the commuting transforms are independent 2D rotations on those planes.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed_round) & ((1 << 64) - 1))

    # π assignment (GQA): Q-head i uses KV-head floor(i * H_KV / H_Q)
    idx = [min((i * H_KV) // H_Q, H_KV - 1) for i in range(H_Q)]
    idx_q = torch.tensor(idx, device=device, dtype=torch.long)

    if rope_aware:
        if d_h % 2 != 0:
            raise ValueError("RoPE-aware mode requires even d_h.")
        half = d_h // 2

        # angles per KV head per plane
        thetas = torch.rand(H_KV, half, generator=g, device="cpu", dtype=torch.float32) * (2 * math.pi)
        cos = torch.cos(thetas).to(device=device, dtype=dtype)
        sin = torch.sin(thetas).to(device=device, dtype=dtype)
        return idx_q, None, cos, sin

    # non-RoPE: arbitrary orthogonal per KV head
    blocks = []
    for _ in range(H_KV):
        S_j = generate_rand_orth_matrix(d_h, g, device="cpu", dtype=torch.float32)  # CPU/FP32
        blocks.append(S_j)

    S_kv = torch.stack(blocks, dim=0).to(device=device, dtype=dtype)  # [H_KV, d_h, d_h]
    return idx_q, S_kv, None, None



# Forward transformation T: W_Q' = W_Q U, W_K' = W_K S, W_V' = W_V S, and W_O' = U^T W_O
# pum_utils.py (NEW)

@torch.no_grad()
def apply_T_attention_weights(
    W_Q: torch.Tensor,  # [H_Q*d_h, d_model]
    W_K: torch.Tensor,  # [H_KV*d_h, d_model]
    W_V: torch.Tensor,  # [H_KV*d_h, d_model]
    W_O: torch.Tensor,  # [d_model, H_Q*d_h]
    idx_q: torch.Tensor,                # [H_Q] long
    S_kv: Optional[torch.Tensor] = None,# [H_KV, d_h, d_h] if non-RoPE
    cos: Optional[torch.Tensor] = None, # [H_KV, d_h//2] if RoPE-aware
    sin: Optional[torch.Tensor] = None, # [H_KV, d_h//2] if RoPE-aware
):
    """
    Implements (PyTorch [out,in] convention):
      W_Q' = U^T W_Q
      W_K' = S^T W_K
      W_V' = S^T W_V
      W_O' = W_O U
    without materializing block-diagonal U or S.
    """
    H_Q = idx_q.numel()
    d_model = W_Q.shape[1]

    if S_kv is not None:
        # ---------- non-RoPE: orthogonal blocks ----------
        H_KV, d_h, _ = S_kv.shape
        assert W_Q.shape[0] == H_Q * d_h
        assert W_K.shape[0] == H_KV * d_h
        assert W_V.shape[0] == H_KV * d_h
        assert W_O.shape[1] == H_Q * d_h

        S_for_Q = S_kv.index_select(0, idx_q)  # [H_Q, d_h, d_h]

        # Q: per-head left multiply by S^T
        WQ = W_Q.reshape(H_Q, d_h, d_model)
        WQp = torch.matmul(S_for_Q.transpose(-1, -2), WQ).reshape(H_Q * d_h, d_model)

        # K/V: per-kv-head left multiply by S^T
        WK = W_K.reshape(H_KV, d_h, d_model)
        WV = W_V.reshape(H_KV, d_h, d_model)
        SKt = S_kv.transpose(-1, -2)
        WKp = torch.matmul(SKt, WK).reshape(H_KV * d_h, d_model)
        WVp = torch.matmul(SKt, WV).reshape(H_KV * d_h, d_model)

        # O: per-Q-head right multiply by S
        WO = W_O.reshape(d_model, H_Q, d_h).permute(1, 0, 2)      # [H_Q, d_model, d_h]
        WOp = torch.matmul(WO, S_for_Q).permute(1, 0, 2).reshape(d_model, H_Q * d_h)

        return WQp, WKp, WVp, WOp

    # ---------- RoPE-aware: split-half plane rotations ----------
    assert cos is not None and sin is not None, "RoPE-aware path requires cos/sin."
    H_KV, half = cos.shape
    d_h = half * 2
    assert W_Q.shape[0] == H_Q * d_h
    assert W_K.shape[0] == H_KV * d_h
    assert W_V.shape[0] == H_KV * d_h
    assert W_O.shape[1] == H_Q * d_h

    cos_q = cos.index_select(0, idx_q)  # [H_Q, half]
    sin_q = sin.index_select(0, idx_q)  # [H_Q, half]

    # Q: W_Q' = U^T W_Q  => apply S^T on (row) pairs
    WQ = W_Q.reshape(H_Q, d_h, d_model)
    A = WQ[:, :half, :]
    B = WQ[:, half:, :]
    c = cos_q.unsqueeze(-1)
    s = sin_q.unsqueeze(-1)
    A_p = c * A + s * B
    B_p = -s * A + c * B
    WQp = torch.cat([A_p, B_p], dim=1).reshape(H_Q * d_h, d_model)

    # K/V: W_K' = S^T W_K, W_V' = S^T W_V
    WK = W_K.reshape(H_KV, d_h, d_model)
    WV = W_V.reshape(H_KV, d_h, d_model)
    A = WK[:, :half, :]; B = WK[:, half:, :]
    c = cos.unsqueeze(-1); s = sin.unsqueeze(-1)
    WKp = torch.cat([c * A + s * B, -s * A + c * B], dim=1).reshape(H_KV * d_h, d_model)

    A = WV[:, :half, :]; B = WV[:, half:, :]
    WVp = torch.cat([c * A + s * B, -s * A + c * B], dim=1).reshape(H_KV * d_h, d_model)

    # O: W_O' = W_O U  => apply S on (column) pairs
    WO = W_O.reshape(d_model, H_Q, d_h).permute(1, 0, 2)  # [H_Q, d_model, d_h]
    A = WO[:, :, :half]
    B = WO[:, :, half:]
    c = cos_q.unsqueeze(1)
    s = sin_q.unsqueeze(1)
    A_p = A * c + B * s
    B_p = -A * s + B * c
    WOp = torch.cat([A_p, B_p], dim=-1).permute(1, 0, 2).reshape(d_model, H_Q * d_h)

    return WQp, WKp, WVp, WOp


# pum_utils.py (NEW)

@torch.no_grad()
def inverse_T_on_update_attention(
    dW_Qp: torch.Tensor,  # [H_Q*d_h, d_model]
    dW_Kp: torch.Tensor,  # [H_KV*d_h, d_model]
    dW_Vp: torch.Tensor,  # [H_KV*d_h, d_model]
    dW_Op: torch.Tensor,  # [d_model, H_Q*d_h]
    idx_q: torch.Tensor,                 # [H_Q] long
    S_kv: Optional[torch.Tensor] = None, # [H_KV, d_h, d_h] if non-RoPE
    cos: Optional[torch.Tensor] = None,  # [H_KV, d_h//2] if RoPE-aware
    sin: Optional[torch.Tensor] = None,  # [H_KV, d_h//2] if RoPE-aware
):
    """
    Inverse on *updates* measured in published coords:
      Δ_Q = U   Δ_Q'
      Δ_K = S   Δ_K'
      Δ_V = S   Δ_V'
      Δ_O = Δ_O' U^T
    """
    H_Q = idx_q.numel()
    d_model = dW_Qp.shape[1]

    if S_kv is not None:
        # ---------- non-RoPE ----------
        H_KV, d_h, _ = S_kv.shape
        S_for_Q = S_kv.index_select(0, idx_q)  # [H_Q, d_h, d_h]

        dQp = dW_Qp.reshape(H_Q, d_h, d_model)
        dQ  = torch.matmul(S_for_Q, dQp).reshape(H_Q * d_h, d_model)

        dKp = dW_Kp.reshape(H_KV, d_h, d_model)
        dVp = dW_Vp.reshape(H_KV, d_h, d_model)
        dK  = torch.matmul(S_kv, dKp).reshape(H_KV * d_h, d_model)
        dV  = torch.matmul(S_kv, dVp).reshape(H_KV * d_h, d_model)

        dOp = dW_Op.reshape(d_model, H_Q, d_h).permute(1, 0, 2)  # [H_Q, d_model, d_h]
        dO  = torch.matmul(dOp, S_for_Q.transpose(-1, -2)).permute(1, 0, 2).reshape(d_model, H_Q * d_h)

        return dQ, dK, dV, dO

    # ---------- RoPE-aware ----------
    assert cos is not None and sin is not None
    H_KV, half = cos.shape
    d_h = half * 2

    cos_q = cos.index_select(0, idx_q)  # [H_Q, half]
    sin_q = sin.index_select(0, idx_q)  # [H_Q, half]

    # Q: dW_Q = S dW_Q'
    dQp = dW_Qp.reshape(H_Q, d_h, d_model)
    A = dQp[:, :half, :]
    B = dQp[:, half:, :]
    c = cos_q.unsqueeze(-1)
    s = sin_q.unsqueeze(-1)
    A_o = c * A - s * B
    B_o = s * A + c * B
    dQ = torch.cat([A_o, B_o], dim=1).reshape(H_Q * d_h, d_model)

    # K/V: dW_K = S dW_K', dW_V = S dW_V'
    dKp = dW_Kp.reshape(H_KV, d_h, d_model)
    dVp = dW_Vp.reshape(H_KV, d_h, d_model)
    c = cos.unsqueeze(-1)
    s = sin.unsqueeze(-1)

    A = dKp[:, :half, :]; B = dKp[:, half:, :]
    dK = torch.cat([c * A - s * B, s * A + c * B], dim=1).reshape(H_KV * d_h, d_model)

    A = dVp[:, :half, :]; B = dVp[:, half:, :]
    dV = torch.cat([c * A - s * B, s * A + c * B], dim=1).reshape(H_KV * d_h, d_model)

    # O: dW_O = dW_O' U^T  => right-multiply by S^T on columns
    dOp = dW_Op.reshape(d_model, H_Q, d_h).permute(1, 0, 2)  # [H_Q, d_model, d_h]
    A = dOp[:, :, :half]
    B = dOp[:, :, half:]
    c = cos_q.unsqueeze(1)
    s = sin_q.unsqueeze(1)
    # right-multiply by S^T: A' = A*c - B*s ; B' = A*s + B*c
    A_o = A * c - B * s
    B_o = A * s + B * c
    dO = torch.cat([A_o, B_o], dim=-1).permute(1, 0, 2).reshape(d_model, H_Q * d_h)

    return dQ, dK, dV, dO



@torch.no_grad()
def apply_T_ffn_weights(
    W1_gate: torch.Tensor,  # [d_ff, d_model]
    W1_up:   torch.Tensor,  # [d_ff, d_model]
    W2:      torch.Tensor,  # [d_model, d_ff]
    perm:    torch.Tensor,  # [d_ff]
    b1_gate: torch.Tensor = None,
    b1_up:   torch.Tensor = None,
    b2_down: torch.Tensor = None,
):
    W1_gate_p = W1_gate.index_select(0, perm)
    W1_up_p   = W1_up.index_select(0, perm)
    W2_p      = W2.index_select(1, perm)

    b1_gate_p = b1_gate.index_select(0, perm) if b1_gate is not None else None
    b1_up_p   = b1_up.index_select(0, perm)   if b1_up   is not None else None
    b2_down_p = b2_down  # unchanged
    return W1_gate_p, W1_up_p, W2_p, b1_gate_p, b1_up_p, b2_down_p


# Inverse transform T^{-1} on FFN parameter updates (Δ)
@torch.no_grad()
def inverse_T_on_update_ffn(
    dW1_gate_p: torch.Tensor,
    dW1_up_p:   torch.Tensor,
    dW2_p:      torch.Tensor,
    inv_perm:   torch.Tensor,  # [d_ff]
    db1_gate_p: torch.Tensor = None,
    db1_up_p:   torch.Tensor = None,
    db2_down_p: torch.Tensor = None,
):
    dW1_gate = dW1_gate_p.index_select(0, inv_perm)
    dW1_up   = dW1_up_p.index_select(0, inv_perm)
    dW2      = dW2_p.index_select(1, inv_perm)

    db1_gate = db1_gate_p.index_select(0, inv_perm) if db1_gate_p is not None else None
    db1_up   = db1_up_p.index_select(0, inv_perm)   if db1_up_p   is not None else None
    db2_down = db2_down_p
    return dW1_gate, dW1_up, dW2, db1_gate, db1_up, db2_down



@torch.no_grad()
def sample_ffn_permutation(d_ff: int, seed: int, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & ((1 << 64) - 1))
    perm = torch.randperm(d_ff, generator=g, device="cpu").to(device)

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(d_ff, device=perm.device)
    return perm, inv




#Old_version
# def sample_ffn_permutation(
#     d_ff: int,
#     seed: int,
#     device: str = "cpu",
#     dtype: torch.dtype = torch.float32,
# ) -> torch.Tensor:
#     """
#     Return the dense permutation matrix P_ffn ∈ R^{d_ff×d_ff} (matrix form). To ensure deterministic
#     behavior, sample the permutation indices on the CPU and then construct P on the target device.
#     """
#     g = torch.Generator(device="cpu")
#     g.manual_seed(int(seed) & ((1 << 64) - 1))
#     idx_cpu = torch.randperm(d_ff, generator=g, device="cpu")
#     idx = idx_cpu.to(device)

#     P = torch.zeros(d_ff, d_ff, device=device, dtype=dtype)
#     P.scatter_(1, idx.view(-1, 1), 1.0)
#     return P


# Memory Saving Version
# @torch.no_grad()
# def apply_T_attention_weights(
#     q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor, o_w: torch.Tensor,
#     S_kv_blocks: List[torch.Tensor],   # len = H_KV, each [d_h, d_h]
#     H_Q: int, H_KV: int, d_h: int
# ):
#     # Q: Left multiplication U^T: multiply S_{π(i)}^T for each Q-head row block
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(q_w)
#         q_w[i*d_h:(i+1)*d_h, :] = S.T @ q_w[i*d_h:(i+1)*d_h, :]

#     # K/V: Left multiply by S_KV^T: Multiply by S_j^T for each row block of the KV-head
#     for j in range(H_KV):
#         S = S_kv_blocks[j].to(k_w)
#         k_w[j*d_h:(j+1)*d_h, :] = S.T @ k_w[j*d_h:(j+1)*d_h, :]
#         v_w[j*d_h:(j+1)*d_h, :] = S.T @ v_w[j*d_h:(j+1)*d_h, :]

#     # O: Right multiply by U: Multiply by U for each Q-head Column block multiplied by S_{π(i)}
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(o_w)
#         o_w[:, i*d_h:(i+1)*d_h] = o_w[:, i*d_h:(i+1)*d_h] @ S


# @torch.no_grad()
# def inverse_T_on_update_attention(
#     dq_p: torch.Tensor, dk_p: torch.Tensor, dv_p: torch.Tensor, do_p: torch.Tensor,
#     S_kv_blocks: List[torch.Tensor],
#     H_Q: int, H_KV: int, d_h: int
# ):
#     dq, dk, dv, do = dq_p.clone(), dk_p.clone(), dv_p.clone(), do_p.clone()

#     # Q: Right multiply by U: Row block multiplication S_{π(i)}
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(dq)
#         dq[i*d_h:(i+1)*d_h, :] = S @ dq[i*d_h:(i+1)*d_h, :]

#     # K/V: Right multiply by S_KV: Row block multiplication S_j
#     for j in range(H_KV):
#         S = S_kv_blocks[j].to(dk)
#         dk[j*d_h:(j+1)*d_h, :] = S @ dk[j*d_h:(j+1)*d_h, :]
#         dv[j*d_h:(j+1)*d_h, :] = S @ dv[j*d_h:(j+1)*d_h, :]

#     # O: Left multiplication U^T: block multiplication S_{π(i)}^T
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(do)
#         do[:, i*d_h:(i+1)*d_h] = do[:, i*d_h:(i+1)*d_h] @ S.T

#     return dq, dk, dv, do


# Harmonic aggregation: aggregate parameter deltas using weights inversely proportional to the provided scaling factors (alphas).
@torch.no_grad()
def harmonic_aggregate(delta_list: List[OrderedDict], alphas: List[float]) -> OrderedDict:
    """
    Compute the harmonic weighted average of m update tensors using weights w_k = 1/alpha_k.
    For numerical stability, all tensors are accumulated in float32 and converted back to their original dtype upon return.
    """
    assert len(delta_list) == len(alphas) and len(delta_list) > 0, \
        f"delta_list({len(delta_list)}) and alphas({len(alphas)}) must have the same non-zero length"

    weights = [1.0 / float(a) for a in alphas]
    S0 = float(sum(weights))

    out = OrderedDict()
    keys = delta_list[0].keys()

    for name in keys:
        v0 = delta_list[0][name]

        if not torch.is_tensor(v0):
            out[name] = v0
            continue

        acc = torch.zeros_like(v0, dtype=torch.float32, device=v0.device)
        
        for i, d in enumerate(delta_list):
            vi = d[name]
            if torch.is_tensor(vi):
                acc.add_(vi.to(torch.float32), alpha=weights[i])

        res = acc / S0
        out[name] = res.to(v0.dtype)

    return out

@torch.no_grad()
def state_dict_l2_norm(sd: "OrderedDict[str, torch.Tensor]") -> float:
    """Compute global L2 norm over all tensor entries in a state_dict."""
    total_sq = None
    for v in sd.values():
        if not torch.is_tensor(v):
            continue
        # 用 float32 计算更稳定；在 v 所在 device 上完成累加，最后再转回 CPU
        s = (v.detach().to(torch.float32).pow(2)).sum()
        total_sq = s if total_sq is None else (total_sq + s)

    if total_sq is None:
        return 0.0
    return float(total_sq.sqrt().detach().cpu())