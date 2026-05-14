# -*- coding: utf-8 -*-
"""
src/trainer/unlearn/test.py

Tests for PUM reparameterization utilities.

We test two things:
1) Functionality preservation: applying reparameterization (attention + FFN) does NOT change model forward outputs.
2) Invertibility: applying reparameterization then applying the inverse recovers original parameters.

This file is intentionally lightweight:
- Uses a tiny randomly initialized Llama model from transformers (no downloads).
- Runs on CPU by default. You can set --device cuda if you want.

Run:
  PYTHONPATH=src python -u src/trainer/unlearn/test.py
or:
  bash scripts/test.sh
"""

import os
import sys
import argparse
from collections import OrderedDict

import torch

# Ensure "src" is on sys.path so "import trainer.*" works even when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                  # .../src/trainer/unlearn
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))         # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from trainer.pum_utils import (  # noqa: E402
    construct_seed64,
    sample_T_attention,
    apply_T_attention_weights,
    inverse_T_on_update_attention,
    sample_ffn_permutation,
    apply_T_ffn_weights,
    inverse_T_on_update_ffn,
)

# -------------------------
# Helpers
# -------------------------

def _fmt_tensor_stats(x: torch.Tensor) -> str:
    with torch.no_grad():
        return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} mean={x.float().mean().item():.4g} std={x.float().std().item():.4g}"


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")

    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if ok:
        return

    diff = (a - b).abs()
    max_abs = diff.max().item()
    denom = b.abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()

    raise AssertionError(
        f"{msg}: NOT allclose (atol={atol}, rtol={rtol}). "
        f"max_abs={max_abs:.6g}, max_rel={max_rel:.6g}\n"
        f"  a: {_fmt_tensor_stats(a)}\n"
        f"  b: {_fmt_tensor_stats(b)}"
    )


def _clone_sd(sd: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    out = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        else:
            out[k] = v
    return out


def _llama_layer_keys(lid: int):
    base_attn = f"model.layers.{lid}.self_attn"
    qk = f"{base_attn}.q_proj.weight"
    kk = f"{base_attn}.k_proj.weight"
    vk = f"{base_attn}.v_proj.weight"
    ok = f"{base_attn}.o_proj.weight"

    base_ffn = f"model.layers.{lid}.mlp"
    gk = f"{base_ffn}.gate_proj.weight"
    uk = f"{base_ffn}.up_proj.weight"
    dk = f"{base_ffn}.down_proj.weight"

    # biases are typically None for Llama, but keep keys for completeness
    gb = f"{base_ffn}.gate_proj.bias"
    ub = f"{base_ffn}.up_proj.bias"
    db = f"{base_ffn}.down_proj.bias"

    return (qk, kk, vk, ok, gk, uk, dk, gb, ub, db)


@torch.no_grad()
def apply_reparam_to_state_dict(
    sd_in: "OrderedDict[str, torch.Tensor]",
    *,
    config,
    rope_aware: bool,
    t_r: int,
    k_copy: int,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Apply PUM reparameterization (attention + FFN) to the given state_dict.
    This is intended to mirror the logic used in PUM.run() for weight reparam.

    Returns a NEW state_dict (cloned tensors).
    """
    sd = _clone_sd(sd_in)

    H_Q = int(getattr(config, "num_attention_heads"))
    H_KV = int(getattr(config, "num_key_value_heads", H_Q))
    d_h = int(getattr(config, "hidden_size")) // H_Q
    L = int(getattr(config, "num_hidden_layers"))

    for lid in range(L):
        (qk, kk, vk, ok, gk, uk, dk, gb, ub, db) = _llama_layer_keys(lid)

        # -------- attention reparam --------
        W_Q = sd.get(qk, None)
        W_K = sd.get(kk, None)
        W_V = sd.get(vk, None)
        W_O = sd.get(ok, None)

        if all(torch.is_tensor(x) for x in [W_Q, W_K, W_V, W_O]):
            seed_layer = construct_seed64(t_r, "T", k_copy, lid)
            idx_q, S_kv, cos, sin = sample_T_attention(
                H_Q, H_KV, d_h,
                rope_aware=rope_aware,
                seed_round=seed_layer,
                k_copy=k_copy,
                device=W_Q.device,
                dtype=W_Q.dtype,
            )
            W_Qp, W_Kp, W_Vp, W_Op = apply_T_attention_weights(
                W_Q, W_K, W_V, W_O,
                idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
            )
            sd[qk], sd[kk], sd[vk], sd[ok] = W_Qp, W_Kp, W_Vp, W_Op

        # -------- FFN reparam (permute hidden channels) --------
        W1g = sd.get(gk, None)
        W1u = sd.get(uk, None)
        W2 = sd.get(dk, None)
        b1g = sd.get(gb, None)
        b1u = sd.get(ub, None)
        b2 = sd.get(db, None)

        if all(torch.is_tensor(x) for x in [W1g, W1u, W2]):
            d_ff = int(W1g.shape[0])
            perm, inv_perm = sample_ffn_permutation(
                d_ff,
                seed=construct_seed64(t_r, "FFN-P", k_copy, lid),
                device=W1g.device,
            )
            W1g_p, W1u_p, W2_p, b1g_p, b1u_p, b2_p = apply_T_ffn_weights(
                W1g, W1u, W2, perm,
                b1_gate=b1g if torch.is_tensor(b1g) else None,
                b1_up=b1u if torch.is_tensor(b1u) else None,
                b2_down=b2 if torch.is_tensor(b2) else None,
            )
            sd[gk], sd[uk], sd[dk] = W1g_p, W1u_p, W2_p
            if torch.is_tensor(b1g): sd[gb] = b1g_p
            if torch.is_tensor(b1u): sd[ub] = b1u_p
            if torch.is_tensor(b2):  sd[db] = b2_p

    return sd


@torch.no_grad()
def inverse_reparam_on_state_dict(
    sd_in: "OrderedDict[str, torch.Tensor]",
    *,
    config,
    rope_aware: bool,
    t_r: int,
    k_copy: int,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Apply the *inverse* of the reparameterization to a given state_dict.

    IMPORTANT:
    We reuse inverse_T_on_update_attention / inverse_T_on_update_ffn (which are written
    for updates) by treating the transformed weights as "updates".
    This matches the algebra: the inverse maps are linear and work on any tensor in the same space.
    """
    sd = _clone_sd(sd_in)

    H_Q = int(getattr(config, "num_attention_heads"))
    H_KV = int(getattr(config, "num_key_value_heads", H_Q))
    d_h = int(getattr(config, "hidden_size")) // H_Q
    L = int(getattr(config, "num_hidden_layers"))

    for lid in range(L):
        (qk, kk, vk, ok, gk, uk, dk, gb, ub, db) = _llama_layer_keys(lid)

        # -------- attention inverse --------
        W_Qp = sd.get(qk, None)
        W_Kp = sd.get(kk, None)
        W_Vp = sd.get(vk, None)
        W_Op = sd.get(ok, None)

        if all(torch.is_tensor(x) for x in [W_Qp, W_Kp, W_Vp, W_Op]):
            seed_layer = construct_seed64(t_r, "T", k_copy, lid)
            idx_q, S_kv, cos, sin = sample_T_attention(
                H_Q, H_KV, d_h,
                rope_aware=rope_aware,
                seed_round=seed_layer,
                k_copy=k_copy,
                device=W_Qp.device,
                dtype=W_Qp.dtype,
            )
            W_Q, W_K, W_V, W_O = inverse_T_on_update_attention(
                W_Qp, W_Kp, W_Vp, W_Op,
                idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
            )
            sd[qk], sd[kk], sd[vk], sd[ok] = W_Q, W_K, W_V, W_O

        # -------- FFN inverse --------
        W1g_p = sd.get(gk, None)
        W1u_p = sd.get(uk, None)
        W2_p = sd.get(dk, None)
        b1g_p = sd.get(gb, None)
        b1u_p = sd.get(ub, None)
        b2_p = sd.get(db, None)

        if all(torch.is_tensor(x) for x in [W1g_p, W1u_p, W2_p]):
            d_ff = int(W1g_p.shape[0])
            perm, inv_perm = sample_ffn_permutation(
                d_ff,
                seed=construct_seed64(t_r, "FFN-P", k_copy, lid),
                device=W1g_p.device,
            )
            W1g, W1u, W2, b1g, b1u, b2 = inverse_T_on_update_ffn(
                W1g_p, W1u_p, W2_p, inv_perm,
                db1_gate_p=b1g_p if torch.is_tensor(b1g_p) else None,
                db1_up_p=b1u_p if torch.is_tensor(b1u_p) else None,
                db2_down_p=b2_p if torch.is_tensor(b2_p) else None,
            )
            sd[gk], sd[uk], sd[dk] = W1g, W1u, W2
            if torch.is_tensor(b1g_p): sd[gb] = b1g
            if torch.is_tensor(b1u_p): sd[ub] = b1u
            if torch.is_tensor(b2_p):  sd[db] = b2

    return sd


# -------------------------
# Tests
# -------------------------

@torch.no_grad()
def test_inverse_attention_and_ffn_matrices(*, rope_aware: bool, device: torch.device, dtype: torch.dtype) -> None:
    """
    Directly test invertibility of the attention/ffn transforms on random matrices.
    """
    torch.manual_seed(0)

    # Small shapes
    d_model = 32
    H_Q = 4
    H_KV = 2
    d_h = d_model // H_Q
    assert d_h * H_Q == d_model

    # Random attention weights
    W_Q = torch.randn(H_Q * d_h, d_model, device=device, dtype=dtype)
    W_K = torch.randn(H_KV * d_h, d_model, device=device, dtype=dtype)
    W_V = torch.randn(H_KV * d_h, d_model, device=device, dtype=dtype)
    W_O = torch.randn(d_model, H_Q * d_h, device=device, dtype=dtype)

    t_r = 123456
    k_copy = 1
    lid = 0
    seed_layer = construct_seed64(t_r, "T", k_copy, lid)

    idx_q, S_kv, cos, sin = sample_T_attention(
        H_Q, H_KV, d_h,
        rope_aware=rope_aware,
        seed_round=seed_layer,
        k_copy=k_copy,
        device=device,
        dtype=dtype,
    )

    W_Qp, W_Kp, W_Vp, W_Op = apply_T_attention_weights(
        W_Q, W_K, W_V, W_O,
        idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
    )
    W_Qr, W_Kr, W_Vr, W_Or = inverse_T_on_update_attention(
        W_Qp, W_Kp, W_Vp, W_Op,
        idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
    )

    _assert_allclose(W_Qr, W_Q, atol=1e-5, rtol=1e-5, msg=f"[inverse][attn][rope_aware={rope_aware}] W_Q")
    _assert_allclose(W_Kr, W_K, atol=1e-5, rtol=1e-5, msg=f"[inverse][attn][rope_aware={rope_aware}] W_K")
    _assert_allclose(W_Vr, W_V, atol=1e-5, rtol=1e-5, msg=f"[inverse][attn][rope_aware={rope_aware}] W_V")
    _assert_allclose(W_Or, W_O, atol=1e-5, rtol=1e-5, msg=f"[inverse][attn][rope_aware={rope_aware}] W_O")

    # Random FFN weights
    d_ff = 64
    W1g = torch.randn(d_ff, d_model, device=device, dtype=dtype)
    W1u = torch.randn(d_ff, d_model, device=device, dtype=dtype)
    W2 = torch.randn(d_model, d_ff, device=device, dtype=dtype)

    perm, inv_perm = sample_ffn_permutation(
        d_ff,
        seed=construct_seed64(t_r, "FFN-P", k_copy, lid),
        device=device,
    )
    W1g_p, W1u_p, W2_p, _, _, _ = apply_T_ffn_weights(W1g, W1u, W2, perm)
    W1g_r, W1u_r, W2_r, _, _, _ = inverse_T_on_update_ffn(W1g_p, W1u_p, W2_p, inv_perm)

    _assert_allclose(W1g_r, W1g, atol=0.0, rtol=0.0, msg="[inverse][ffn] W1_gate (perm should be exact)")
    _assert_allclose(W1u_r, W1u, atol=0.0, rtol=0.0, msg="[inverse][ffn] W1_up   (perm should be exact)")
    _assert_allclose(W2_r,  W2,  atol=0.0, rtol=0.0, msg="[inverse][ffn] W2_down (perm should be exact)")


@torch.no_grad()
def test_state_dict_inverse_and_function_preservation(
    *,
    rope_aware: bool,
    device: torch.device,
    dtype: torch.dtype,
    atol_logits: float,
    rtol_logits: float,
) -> None:
    """
    End-to-end test on a tiny Llama model:
    1) forward logits invariant under weight reparameterization
    2) inverse(T(sd)) recovers original sd on the transformed keys
    """
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as e:
        raise RuntimeError(
            "transformers is required for this test. "
            "Please install transformers in your env."
        ) from e

    torch.manual_seed(0)

    # Tiny config (no downloads)
    cfg = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,     # test GQA path too
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()

    bs, seqlen = 2, 16
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(bs, seqlen), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        out0 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits0 = out0.logits.detach()

    sd0 = OrderedDict((k, v.detach().clone() if torch.is_tensor(v) else v) for k, v in model.state_dict().items())

    t_r = 202000
    k_copy = 1

    # Apply reparameterization to weights
    sdT = apply_reparam_to_state_dict(sd0, config=cfg, rope_aware=rope_aware, t_r=t_r, k_copy=k_copy)

    # Load into a new model
    modelT = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    modelT.load_state_dict(sdT, strict=True)
    modelT.eval()

    with torch.inference_mode():
        outT = modelT(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logitsT = outT.logits.detach()

    _assert_allclose(
        logitsT, logits0,
        atol=atol_logits, rtol=rtol_logits,
        msg=f"[function-preserving][rope_aware={rope_aware}] logits"
    )

    # Apply inverse to recover sd
    sdR = inverse_reparam_on_state_dict(sdT, config=cfg, rope_aware=rope_aware, t_r=t_r, k_copy=k_copy)

    # Only compare the keys we actually transform (per layer attention + ffn weights/biases)
    for lid in range(cfg.num_hidden_layers):
        keys = _llama_layer_keys(lid)
        for name in keys:
            if name in sd0 and torch.is_tensor(sd0[name]) and name in sdR and torch.is_tensor(sdR[name]):
                _assert_allclose(
                    sdR[name], sd0[name],
                    atol=1e-5, rtol=1e-5,
                    msg=f"[inverse][rope_aware={rope_aware}] state_dict[{name}]"
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=os.environ.get("PUM_TEST_DEVICE", "cpu"),
                   help="cpu or cuda (default: cpu). You may also set env PUM_TEST_DEVICE.")
    p.add_argument("--dtype", type=str, default=os.environ.get("PUM_TEST_DTYPE", "float32"),
                   help="float32/float16/bfloat16 (default: float32).")
    p.add_argument("--rope", type=str, default=os.environ.get("PUM_TEST_ROPE", "both"),
                   choices=["true", "false", "both"],
                   help="Test rope_aware=true/false/both (default: both).")
    p.add_argument("--atol_logits", type=float, default=float(os.environ.get("PUM_TEST_ATOL", "1e-4")))
    p.add_argument("--rtol_logits", type=float, default=float(os.environ.get("PUM_TEST_RTOL", "1e-4")))
    args = p.parse_args()

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype={args.dtype}, choose from {list(dtype_map.keys())}")
    dtype = dtype_map[args.dtype]

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")

    rope_cases = []
    if args.rope == "both":
        rope_cases = [False, True]
    elif args.rope == "true":
        rope_cases = [True]
    else:
        rope_cases = [False]

    print("============================================================")
    print(f"[PUM-TEST] device={device} dtype={dtype} rope_cases={rope_cases}")
    print(f"[PUM-TEST] logits tolerance: atol={args.atol_logits} rtol={args.rtol_logits}")
    print("============================================================", flush=True)

    # 1) Matrix-level inversion tests (attention + ffn)
    for rope_aware in rope_cases:
        print(f"\n[TEST] inverse matrices (rope_aware={rope_aware})", flush=True)
        test_inverse_attention_and_ffn_matrices(rope_aware=rope_aware, device=device, dtype=dtype)
        print(f"[PASS] inverse matrices (rope_aware={rope_aware})", flush=True)

    # 2) Model-level function preservation + inverse(T(sd)) recovery tests
    for rope_aware in rope_cases:
        print(f"\n[TEST] model forward invariance + sd inverse (rope_aware={rope_aware})", flush=True)
        test_state_dict_inverse_and_function_preservation(
            rope_aware=rope_aware,
            device=device,
            dtype=dtype,
            atol_logits=args.atol_logits,
            rtol_logits=args.rtol_logits,
        )
        print(f"[PASS] model forward invariance + sd inverse (rope_aware={rope_aware})", flush=True)

    print(" All PUM reparameterization tests PASSED.", flush=True)


if __name__ == "__main__":
    main()