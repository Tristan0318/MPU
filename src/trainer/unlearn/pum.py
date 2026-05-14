# -*- coding: utf-8 -*-
# src/trainer/unlearn/pum.py

import copy
import gc
import os
import torch
from transformers import AutoModelForCausalLM
from trainer.unlearn.base import UnlearnTrainer

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import List, Optional, Literal
from omegaconf import DictConfig, OmegaConf
from trainer.pum_utils import (
    construct_seed64,
    compute_layerwise_sigma,
    generate_zero_sum_noises,
    sample_T_attention,
    apply_T_attention_weights,
    inverse_T_on_update_attention,
    sample_ffn_permutation,
    apply_T_ffn_weights,
    inverse_T_on_update_ffn,
    harmonic_aggregate,
    state_dict_l2_norm,
)

from trainer.utils import (
    compute_gradascent, 
    compute_graddiff,
    compute_dpo,
    compute_npo,
    compute_simnpo,
    compute_satimp,
    compute_undial,
    compute_wga,
    prepare_ref_model,
)


@dataclass
class PUMConfig:
    # Public base model for task vector sigma
    base_model_name_or_path: Optional[str] = None

    # Server parameters
    R: int = 5
    m: int = 3
    alphas: Optional[List[float]] = None
    kappa: float = 0.1
    eta_srv: float = 1.0
    rope_aware: bool = True

    # Residual/norm global permutation
    use_residual_perm: bool = False

    # Client local unlearning algorithms
    client_method: Literal["GradAscent", "GradDiff", "DPO", "NPO", "SimNPO", "SatImp", "UnDIAL", "WGA"] = "GradAscent"

    # Epoch-Based Inner Loop
    client_round_epoch: Optional[int] = None

    # Legacy: step-based inner loop fallback (kept for backward compatibility)
    client_steps: int = 10

    client_lr: float = 1e-5

    # Deterministic seeds
    s_noise: Optional[List[int]] = None
    t_reparam: Optional[List[int]] = None
    reparam: bool = True


class PUM(UnlearnTrainer):
    def __init__(self, model=None, tokenizer=None, cfg: PUMConfig = None, *args, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)

        # Robust Hydra/OmegaConf
        if cfg is None:
            cfg = PUMConfig()
        elif not isinstance(cfg, PUMConfig):
            # Accept DictConfig / dict
            if OmegaConf.is_config(cfg):
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            elif isinstance(cfg, dict):
                cfg_dict = cfg
            else:
                cfg_dict = dict(cfg)

            valid = {f.name for f in fields(PUMConfig)}
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
            cfg = PUMConfig(**cfg_dict)

        self.model = model
        self.tok = tokenizer
        self.cfg = cfg

        # Cache for pretrained base state dict on CPU
        self._base_sd_cpu: Optional[OrderedDict] = None

    # CUDA memory debug helpers
    def _mem_debug_level(self) -> int:
        v = os.environ.get("PUM_MEM_DEBUG", "0")
        try:
            return int(v)
        except Exception:
            return 1 if str(v).lower() in {"1", "true", "yes", "y"} else 0

    def _cuda_mem(self, tag: str, device) -> None:
        if self._mem_debug_level() <= 0:
            return
        if not torch.cuda.is_available():
            return

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type != "cuda":
            return

        torch.cuda.synchronize(dev)
        free, total = torch.cuda.mem_get_info(dev)
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        max_alloc = torch.cuda.max_memory_allocated(dev)
        max_reserved = torch.cuda.max_memory_reserved(dev)

        print(
            f"[MEM][{tag}] free={free/2**20:.1f}MB "
            f"alloc={alloc/2**20:.1f}MB reserved={reserved/2**20:.1f}MB "
            f"max_alloc={max_alloc/2**20:.1f}MB max_reserved={max_reserved/2**20:.1f}MB",
            flush=True,
        )

    def _cuda_mem_summary(self, device, tag: str = "") -> None:
        if self._mem_debug_level() <= 0:
            return
        if not torch.cuda.is_available():
            return
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type != "cuda":
            return

        print(f"[MEM][summary]{(' '+tag) if tag else ''}", flush=True)
        print(torch.cuda.memory_summary(dev, abbreviated=True), flush=True)

    def alphas(self) -> List[float]:
        if self.cfg.alphas is not None:
            assert len(self.cfg.alphas) == self.cfg.m
            return self.cfg.alphas
        xs = [1.0]
        if self.cfg.m>1:
            for num_copy in range(1, self.cfg.m):
                xs.append(1+1/(self.cfg.m-1)*num_copy)
        return xs

    @torch.no_grad()
    def layer_groups(self) -> dict:
        sd = self.model.state_dict()
        groups = {}
        for k in sd.keys():
            if not k.startswith("model.layers."):
                continue
            parts = k.split(".")
            if len(parts) > 3:
                lid = parts[2]
                groups.setdefault(f"layer{lid}", []).append(k)
        return groups

    @torch.no_grad()
    def _load_base_sd(self, layer_groups: dict) -> OrderedDict:
        """
        Load the *public pretrained* reference model state_dict (CPU),
        filtered to only the keys needed for sigma computation (model.layers.*).
        """
        if self._base_sd_cpu is not None:
            return self._base_sd_cpu

        base_path = self.cfg.base_model_name_or_path
        if base_path is None:
            raise ValueError("PUMConfig.base_model_name_or_path is None. Please set it in configs/trainer/PUM.yaml")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base_sd_full = base_model.state_dict()

        need = set()
        for names in layer_groups.values():
            need.update(names)

        base_sd = OrderedDict()
        missing = []
        for name in need:
            if name not in base_sd_full:
                missing.append(name)
                continue
            t = base_sd_full[name]
            if torch.is_tensor(t):
                base_sd[name] = t.detach().cpu().to(torch.float32).clone()

        if len(missing) > 0:
            print(f"[PUM][Warning] {len(missing)} keys missing in base model state_dict. Example: {missing[:5]}")

        del base_model
        self._base_sd_cpu = base_sd
        return base_sd


    def client_unlearn(
        self,
        ul_model,
        ul_forget,
        ul_retain=None,
        ref_model=None,
        device="cuda",
        mem_tag: str = "client",
        *,
        epoch_offset: int = 0,
        count_global_step: bool = False,
        ):
        """
        Client inner loop.

        If cfg.client_round_epoch is set:
        - iterate over ul_forget for (client_round_epoch) epochs
        - use HF-style gradient accumulation (self.args.gradient_accumulation_steps)

        If cfg.client_round_epoch is None:
        - fallback to legacy step-based loop (cfg.client_steps optimizer updates)

        count_global_step:
        - if True, increment self.state.global_step at every optimizer update
            (use it only on k==1 to match baseline global_step while total compute is m-times)
        """

        import math

        def to_device_batch(batch, dev):
            return {k: v.to(dev) for k, v in batch.items()}

        def next_from(it, dl):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            return b, it

        # HF args: gradient accumulation & max grad norm
        grad_accum = int(getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1) or 1)
        if grad_accum < 1:
            grad_accum = 1
        max_grad_norm = float(getattr(getattr(self, "args", None), "max_grad_norm", 1.0) or 1.0)

        # Disable KV cache for training stability/memory
        orig_use_cache = getattr(getattr(ul_model, "config", None), "use_cache", None)
        if orig_use_cache is not None:
            ul_model.config.use_cache = False

        ul_model.to(device)
        self._cuda_mem(f"{mem_tag}/begin", device)
        ul_model.train()

        # AdamW: foreach=False helps avoid extra memory spikes
        try:
            optim = torch.optim.AdamW(ul_model.parameters(), lr=self.cfg.client_lr, foreach=False)
        except TypeError:
            optim = torch.optim.AdamW(ul_model.parameters(), lr=self.cfg.client_lr)

        method = str(self.cfg.client_method)
        retain_loss_type = getattr(self.cfg, "retain_loss_type", "NLL")

        alpha = getattr(self.cfg, "retain_alpha", 1.0)
        gamma = getattr(self.cfg, "forget_gamma", 1.0)

        dpo_beta    = getattr(self.cfg, "dpo_beta", 0.1)
        npo_beta    = getattr(self.cfg, "npo_beta", dpo_beta)
        undial_beta = getattr(self.cfg, "undial_beta", 10.0)

        simnpo_beta  = getattr(self.cfg, "simnpo_beta", 4.5)
        simnpo_delta = getattr(self.cfg, "simnpo_delta", 0.0)

        wga_beta     = getattr(self.cfg, "wga_beta", 1.0)
        satimp_beta1 = getattr(self.cfg, "satimp_beta1", 5.0)
        satimp_beta2 = getattr(self.cfg, "satimp_beta2", 1.0)

        need_retain = method in {"GradDiff", "NPO", "SimNPO", "SatImp", "UnDIAL", "WGA", "DPO"}

        # Teacher requirement
        need_ref = (method in {"DPO", "NPO", "UnDIAL"}) or (
            retain_loss_type == "KL" and method in {"GradDiff", "WGA", "SimNPO", "SatImp"}
        )
        teacher = ref_model
        if teacher is None and need_ref:
            teacher = prepare_ref_model(ul_model, device=device)

        # Epoch-based mode
        if self.cfg.client_round_epoch is not None:
            round_epochs = int(self.cfg.client_round_epoch)
            if round_epochs < 1:
                round_epochs = 1

            last = None

            for ep in range(round_epochs):
                # If distributed sampler exists, set epoch for deterministic shuffling
                try:
                    if hasattr(ul_forget, "sampler") and hasattr(ul_forget.sampler, "set_epoch"):
                        ul_forget.sampler.set_epoch(epoch_offset + ep)
                except Exception:
                    pass
                try:
                    if ul_retain is not None and hasattr(ul_retain, "sampler") and hasattr(ul_retain.sampler, "set_epoch"):
                        ul_retain.sampler.set_epoch(epoch_offset + ep)
                except Exception:
                    pass

                # retain iterator (cycled as needed)
                it_r = iter(ul_retain) if ul_retain is not None else None

                accum = 0
                optim.zero_grad(set_to_none=True)

                for step_idx, bf in enumerate(ul_forget):
                    if self._mem_debug_level() >= 1 and ep == 0 and step_idx == 0:
                        self._cuda_mem(f"{mem_tag}/ep0/step0/before_forward", device)

                    # Build inputs
                    if method == "DPO":
                        assert isinstance(bf, dict) and ("original" in bf and "alternate" in bf), \
                            "DPO requires {'original': batch, 'alternate': batch}"
                        assert ul_retain is not None, "DPO requires retain dataloader"

                        bf_inputs = {
                            "original": to_device_batch(bf["original"], device),
                            "alternate": to_device_batch(bf["alternate"], device),
                        }
                        br, it_r = next_from(it_r, ul_retain)
                        br_inputs = to_device_batch(br, device)
                        inputs = {"forget": bf_inputs, "retain": br_inputs}
                    else:
                        bf_inputs = to_device_batch(bf, device)
                        if need_retain and method != "GradAscent":
                            assert ul_retain is not None, f"{method} requires retain dataloader"
                            br, it_r = next_from(it_r, ul_retain)
                            br_inputs = to_device_batch(br, device)
                            inputs = {"forget": bf_inputs, "retain": br_inputs}
                        else:
                            inputs = {"forget": bf_inputs}

                    # (optional) print batch shape once for debugging
                    if self._mem_debug_level() >= 2 and ep == 0 and step_idx == 0:
                        try:
                            sample = bf_inputs if isinstance(bf_inputs, dict) else bf_inputs.get("original", {})
                            for kk in ["input_ids", "labels", "attention_mask"]:
                                if kk in sample and torch.is_tensor(sample[kk]):
                                    t = sample[kk]
                                    print(
                                        f"[BATCH][{mem_tag}] {kk}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}",
                                        flush=True,
                                    )
                        except Exception:
                            pass

                    # Forward (compute loss)
                    try:
                        if method == "GradAscent":
                            loss, _ = compute_gradascent(ul_model, inputs)
                        elif method == "GradDiff":
                            loss, _ = compute_graddiff(
                                ul_model, inputs,
                                gamma=gamma, alpha=alpha,
                                retain_loss_type=retain_loss_type,
                            )
                        elif method == "DPO":
                            loss, _ = compute_dpo(
                                ul_model, inputs,
                                alpha=alpha, beta=dpo_beta, gamma=gamma,
                                retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                            )
                        elif method == "SimNPO":
                            loss, _ = compute_simnpo(
                                ul_model, inputs,
                                alpha=alpha, beta=simnpo_beta, delta=simnpo_delta, gamma=gamma,
                                retain_loss_type=retain_loss_type,
                            )
                        elif method == "NPO":
                            loss, _ = compute_npo(
                                ul_model, inputs,
                                alpha=alpha, beta=npo_beta, gamma=gamma,
                                retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                            )
                        elif method == "UnDIAL":
                            loss, _ = compute_undial(
                                ul_model, inputs,
                                alpha=alpha, beta=undial_beta, gamma=gamma,
                                retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                            )
                        elif method == "WGA":
                            loss, _ = compute_wga(
                                ul_model, inputs,
                                alpha=alpha, beta=wga_beta, gamma=gamma,
                                retain_loss_type=retain_loss_type,
                            )
                        elif method == "SatImp":
                            loss, _ = compute_satimp(
                                ul_model, inputs,
                                alpha=alpha, beta1=satimp_beta1, beta2=satimp_beta2, gamma=gamma,
                                retain_loss_type=retain_loss_type,
                            )
                        else:
                            raise NotImplementedError(f"unknown client_method={self.cfg.client_method}")
                    except torch.cuda.OutOfMemoryError:
                        self._cuda_mem(f"{mem_tag}/ep{ep}/step{step_idx}/OOM_in_forward", device)
                        self._cuda_mem_summary(device, tag=f"{mem_tag}/ep{ep}/forward")
                        raise

                    # Gradient accumulation: match HF semantics
                    try:
                        (loss / float(grad_accum)).backward()
                    except torch.cuda.OutOfMemoryError:
                        self._cuda_mem(f"{mem_tag}/ep{ep}/step{step_idx}/OOM_in_backward", device)
                        self._cuda_mem_summary(device, tag=f"{mem_tag}/ep{ep}/backward")
                        raise

                    last = float(loss.detach().cpu())
                    accum += 1

                    # Update step
                    if accum >= grad_accum:
                        torch.nn.utils.clip_grad_norm_(ul_model.parameters(), max_grad_norm)
                        try:
                            optim.step()
                        except torch.cuda.OutOfMemoryError:
                            self._cuda_mem(f"{mem_tag}/ep{ep}/step{step_idx}/OOM_in_optim_step", device)
                            self._cuda_mem_summary(device, tag=f"{mem_tag}/ep{ep}/optim")
                            raise
                        optim.zero_grad(set_to_none=True)
                        accum = 0

                        # global_step (口径_2): count only when enabled
                        if count_global_step and hasattr(self, "state"):
                            self.state.global_step += 1

                # Flush remainder at epoch end (HF-style)
                if accum > 0:
                    torch.nn.utils.clip_grad_norm_(ul_model.parameters(), max_grad_norm)
                    try:
                        optim.step()
                    except torch.cuda.OutOfMemoryError:
                        self._cuda_mem(f"{mem_tag}/ep{ep}/OOM_in_optim_step_flush", device)
                        self._cuda_mem_summary(device, tag=f"{mem_tag}/ep{ep}/optim_flush")
                        raise
                    optim.zero_grad(set_to_none=True)
                    accum = 0
                    if count_global_step and hasattr(self, "state"):
                        self.state.global_step += 1

            if orig_use_cache is not None:
                ul_model.config.use_cache = orig_use_cache

            ul_model.zero_grad(set_to_none=True)
            del optim
            gc.collect()
            return last

        # Legacy step-based mode
        it_f = iter(ul_forget)
        it_r = iter(ul_retain) if ul_retain is not None else None

        last = None
        for step in range(self.cfg.client_steps):
            bf, it_f = next_from(it_f, ul_forget)

            if method == "DPO":
                assert isinstance(bf, dict) and ("original" in bf and "alternate" in bf), \
                    "DPO requires {'original': batch, 'alternate': batch}"
                assert ul_retain is not None, "DPO requires retain dataloader"
                bf_inputs = {
                    "original": to_device_batch(bf["original"], device),
                    "alternate": to_device_batch(bf["alternate"], device),
                }
                br, it_r = next_from(it_r, ul_retain)
                br_inputs = to_device_batch(br, device)
                inputs = {"forget": bf_inputs, "retain": br_inputs}
            else:
                bf_inputs = to_device_batch(bf, device)
                if need_retain and method != "GradAscent":
                    assert ul_retain is not None, f"{method} requires retain dataloader"
                    br, it_r = next_from(it_r, ul_retain)
                    br_inputs = to_device_batch(br, device)
                    inputs = {"forget": bf_inputs, "retain": br_inputs}
                else:
                    inputs = {"forget": bf_inputs}

            try:
                if method == "GradAscent":
                    loss, _ = compute_gradascent(ul_model, inputs)
                elif method == "GradDiff":
                    loss, _ = compute_graddiff(
                        ul_model, inputs,
                        gamma=gamma, alpha=alpha,
                        retain_loss_type=retain_loss_type,
                    )
                elif method == "DPO":
                    loss, _ = compute_dpo(
                        ul_model, inputs,
                        alpha=alpha, beta=dpo_beta, gamma=gamma,
                        retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                    )
                elif method == "SimNPO":
                    loss, _ = compute_simnpo(
                        ul_model, inputs,
                        alpha=alpha, beta=simnpo_beta, delta=simnpo_delta, gamma=gamma,
                        retain_loss_type=retain_loss_type,
                    )
                elif method == "NPO":
                    loss, _ = compute_npo(
                        ul_model, inputs,
                        alpha=alpha, beta=npo_beta, gamma=gamma,
                        retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                    )
                elif method == "UnDIAL":
                    loss, _ = compute_undial(
                        ul_model, inputs,
                        alpha=alpha, beta=undial_beta, gamma=gamma,
                        retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                    )
                elif method == "WGA":
                    loss, _ = compute_wga(
                        ul_model, inputs,
                        alpha=alpha, beta=wga_beta, gamma=gamma,
                        retain_loss_type=retain_loss_type,
                    )
                elif method == "SatImp":
                    loss, _ = compute_satimp(
                        ul_model, inputs,
                        alpha=alpha, beta1=satimp_beta1, beta2=satimp_beta2, gamma=gamma,
                        retain_loss_type=retain_loss_type,
                    )
                else:
                    raise NotImplementedError(f"unknown client_method={self.cfg.client_method}")
            except torch.cuda.OutOfMemoryError:
                self._cuda_mem(f"{mem_tag}/step{step}/OOM_in_forward", device)
                self._cuda_mem_summary(device, tag=f"{mem_tag}/step{step}/forward")
                raise

            optim.zero_grad(set_to_none=True)
            try:
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                self._cuda_mem(f"{mem_tag}/step{step}/OOM_in_backward", device)
                self._cuda_mem_summary(device, tag=f"{mem_tag}/step{step}/backward")
                raise

            torch.nn.utils.clip_grad_norm_(ul_model.parameters(), max_grad_norm)
            try:
                optim.step()
            except torch.cuda.OutOfMemoryError:
                self._cuda_mem(f"{mem_tag}/step{step}/OOM_in_optim_step", device)
                self._cuda_mem_summary(device, tag=f"{mem_tag}/step{step}/optim")
                raise

            last = float(loss.detach().cpu())
            if count_global_step and hasattr(self, "state"):
                self.state.global_step += 1

        if orig_use_cache is not None:
            ul_model.config.use_cache = orig_use_cache

        ul_model.zero_grad(set_to_none=True)
        del optim
        gc.collect()
        return last

    def run(self, ul_forget, ul_retain=None, ref_model=None, device="cuda", base_sd: OrderedDict = None):
        self.model.to(device).eval()

        print(f"[PUM][MEM_DEBUG] level={self._mem_debug_level()} device={device}", flush=True)
        try:
            p = next(self.model.parameters())
            print(f"[PUM] param dtype={p.dtype} param device={p.device}", flush=True)
        except Exception:
            pass

        # Baseline-aligned bookkeeping prints
        num_train_epochs = int(getattr(getattr(self, "args", None), "num_train_epochs", 10) or 10)
        grad_accum = int(getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1) or 1)
        if grad_accum < 1:
            grad_accum = 1

        client_round_epoch = self.cfg.client_round_epoch
        if client_round_epoch is not None:
            client_round_epoch = int(client_round_epoch)
            if client_round_epoch < 1:
                client_round_epoch = 1
            total_epochs_actual = int(self.cfg.R) * int(client_round_epoch)
            if total_epochs_actual != num_train_epochs:
                print(
                    f"[PUM][WARN] R({self.cfg.R}) * client_round_epoch({client_round_epoch}) = {total_epochs_actual} "
                    f"!= trainer.args.num_train_epochs({num_train_epochs}). Continue anyway.",
                    flush=True,
                )

        # Expected baseline global_step from *actual dataloader length*
        # updates_per_epoch = ceil(len(ul_forget) / grad_accum)
        try:
            import math
            micro_steps_per_epoch = len(ul_forget)
            updates_per_epoch = int(math.ceil(float(micro_steps_per_epoch) / float(grad_accum)))
            expected_baseline_global_step = updates_per_epoch * num_train_epochs
            print(
                f"[PUM][STEP-CHECK] len(ul_forget)={micro_steps_per_epoch} micro-steps/epoch, "
                f"grad_acc={grad_accum} => updates/epoch={updates_per_epoch}. "
                f"baseline num_train_epochs={num_train_epochs} => expected global_step={expected_baseline_global_step}.",
                flush=True,
            )
        except Exception:
            print(
                "[PUM][STEP-CHECK] Could not infer expected global_step from len(ul_forget). "
                "This is normal if the dataloader has no __len__().",
                flush=True,
            )

        if hasattr(self, "state"):
            print(f"[PUM][STEP-CHECK] entering run(): current self.state.global_step={self.state.global_step}", flush=True)

        # Memory stats
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        self._cuda_mem("run/start", device)

        theta_prev: OrderedDict = copy.deepcopy(self.model.state_dict())
        self._cuda_mem("run/after_theta_prev", device)

        layer_groups = self.layer_groups()
        alphas = self.alphas()

        if base_sd is None:
            base_sd = self._load_base_sd(layer_groups)

        theta_norm = state_dict_l2_norm(theta_prev)
        print(f"[Debug] Init ||θ||₂ = {theta_norm:.6f}")

        H_Q = getattr(self.model.config, "num_attention_heads", 32)
        H_KV = getattr(self.model.config, "num_key_value_heads", H_Q)
        d_h = getattr(self.model.config, "hidden_size", 4096) // H_Q
        L = getattr(self.model.config, "num_hidden_layers", 1)

        for r in range(1, self.cfg.R + 1):
            self._cuda_mem(f"r{r}/start", device)
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(device))

            s_r = (self.cfg.s_noise[r - 1] if self.cfg.s_noise else 101_000 + r)
            t_r = (self.cfg.t_reparam[r - 1] if self.cfg.t_reparam else 202_000 + r)

            sig = compute_layerwise_sigma(theta_prev, layer_groups, self.cfg.kappa, base_sd=base_sd)
            self._cuda_mem(f"r{r}/after_sigma", device)

            noises = generate_zero_sum_noises(
                theta_prev, layer_groups, sig, self.cfg.m, alphas, s_r, torch.device(device)
            )
            self._cuda_mem(f"r{r}/after_noises", device)

            S0 = 0.0
            S1: OrderedDict = OrderedDict()

            # Epoch_offset for this round (for continuous shuffling like baseline epochs)
            epoch_offset_round = 0
            if client_round_epoch is not None:
                epoch_offset_round = (r - 1) * int(client_round_epoch)

            for k in range(1, self.cfg.m + 1):
                self._cuda_mem(f"r{r}/k{k}/before_pub_sd", device)

                pub_sd = copy.deepcopy(theta_prev)
                self._cuda_mem(f"r{r}/k{k}/after_pub_sd", device)

                noise_k = noises[k - 1]
                for name, t in pub_sd.items():
                    if torch.is_tensor(t) and (name in noise_k):
                        t.add_(noise_k[name])
                noises[k - 1] = None
                del noise_k
                self._cuda_mem(f"r{r}/k{k}/after_noise_add", device)

                # apply reparam
                if self.cfg.reparam != False:
                    for lid in range(L):
                        base_attn = f"model.layers.{lid}.self_attn"
                        qk = f"{base_attn}.q_proj.weight"; kk = f"{base_attn}.k_proj.weight"
                        vk = f"{base_attn}.v_proj.weight"; ok = f"{base_attn}.o_proj.weight"
                        W_Q = pub_sd.get(qk, None)
                        W_K = pub_sd.get(kk, None)
                        W_V = pub_sd.get(vk, None)
                        W_O = pub_sd.get(ok, None)

                        if all(t is not None for t in [W_Q, W_K, W_V, W_O]):
                            seed_layer = construct_seed64(t_r, "T", k, lid)
                            idx_q, S_kv, cos, sin = sample_T_attention(
                                H_Q, H_KV, d_h, self.cfg.rope_aware,
                                seed_round=seed_layer, k_copy=k,
                                device=W_Q.device, dtype=W_Q.dtype
                            )
                            W_Qp, W_Kp, W_Vp, W_Op = apply_T_attention_weights(
                                W_Q, W_K, W_V, W_O,
                                idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
                            )
                            pub_sd[qk], pub_sd[kk], pub_sd[vk], pub_sd[ok] = W_Qp, W_Kp, W_Vp, W_Op

                        base_ffn = f"model.layers.{lid}.mlp"
                        gk = f"{base_ffn}.gate_proj.weight"; uk = f"{base_ffn}.up_proj.weight"
                        dk = f"{base_ffn}.down_proj.weight"
                        gb = f"{base_ffn}.gate_proj.bias";  ub = f"{base_ffn}.up_proj.bias";  db = f"{base_ffn}.down_proj.bias"

                        W1g = pub_sd.get(gk); W1u = pub_sd.get(uk); W2 = pub_sd.get(dk)
                        b1g = pub_sd.get(gb); b1u = pub_sd.get(ub); b2 = pub_sd.get(db)

                        if (W1g is not None) and (W1u is not None) and (W2 is not None):
                            d_ff = W1g.shape[0]
                            perm, inv_perm = sample_ffn_permutation(
                                d_ff,
                                seed=construct_seed64(t_r, "FFN-P", k, lid),
                                device=W1g.device,
                            )
                            W1g_p, W1u_p, W2_p, b1g_p, b1u_p, b2_p = apply_T_ffn_weights(
                                W1g, W1u, W2, perm, b1_gate=b1g, b1_up=b1u, b2_down=b2
                            )
                            pub_sd[gk], pub_sd[uk], pub_sd[dk] = W1g_p, W1u_p, W2_p
                            if b1g is not None: pub_sd[gb] = b1g_p
                            if b1u is not None: pub_sd[ub] = b1u_p
                            if b2  is not None: pub_sd[db] = b2_p

                    self._cuda_mem(f"r{r}/k{k}/after_reparam", device)

                # client update
                self._cuda_mem(f"r{r}/k{k}/before_model_copy", device)
                model_k = copy.deepcopy(self.model).to(device)
                model_k.load_state_dict(pub_sd, strict=False)
                self._cuda_mem(f"r{r}/k{k}/after_load_state", device)

                self._cuda_mem(f"r{r}/k{k}/before_client", device)
                try:
                    _ = self.client_unlearn(
                        model_k, ul_forget, ul_retain, ref_model,
                        device=device,
                        mem_tag=f"r{r}/k{k}/client",
                        epoch_offset=epoch_offset_round,
                        count_global_step=(k == 1),  # baseline-aligned global_step
                    )
                except torch.cuda.OutOfMemoryError:
                    self._cuda_mem(f"r{r}/k{k}/OOM_in_client_unlearn", device)
                    self._cuda_mem_summary(device, tag=f"r{r}/k{k}")
                    raise
                self._cuda_mem(f"r{r}/k{k}/after_client", device)

                # Delta' and inverse
                after_sd = model_k.state_dict()
                delta_prime = OrderedDict()
                for name in pub_sd.keys():
                    if torch.is_tensor(pub_sd[name]):
                        delta_prime[name] = after_sd[name] - pub_sd[name]
                if self.cfg.reparam != False:
                    for lid in range(L):
                        base_attn = f"model.layers.{lid}.self_attn"
                        qk = f"{base_attn}.q_proj.weight"; kk = f"{base_attn}.k_proj.weight"
                        vk = f"{base_attn}.v_proj.weight"; ok = f"{base_attn}.o_proj.weight"
                        dQp = delta_prime.get(qk, None)
                        dKp = delta_prime.get(kk, None)
                        dVp = delta_prime.get(vk, None)
                        dOp = delta_prime.get(ok, None)

                        if all(t is not None for t in [dQp, dKp, dVp, dOp]):
                            seed_layer = construct_seed64(t_r, "T", k, lid)
                            idx_q, S_kv, cos, sin = sample_T_attention(
                                H_Q, H_KV, d_h, self.cfg.rope_aware,
                                seed_round=seed_layer, k_copy=k,
                                device=dQp.device, dtype=dQp.dtype
                            )
                            dQ, dK, dV, dO = inverse_T_on_update_attention(
                                dQp, dKp, dVp, dOp,
                                idx_q=idx_q, S_kv=S_kv, cos=cos, sin=sin
                            )
                            delta_prime[qk], delta_prime[kk], delta_prime[vk], delta_prime[ok] = dQ, dK, dV, dO

                        base_ffn = f"model.layers.{lid}.mlp"
                        gk = f"{base_ffn}.gate_proj.weight"; uk = f"{base_ffn}.up_proj.weight"
                        dk = f"{base_ffn}.down_proj.weight"
                        gb = f"{base_ffn}.gate_proj.bias";  ub = f"{base_ffn}.up_proj.bias";  db = f"{base_ffn}.down_proj.bias"

                        dW1g_p = delta_prime.get(gk); dW1u_p = delta_prime.get(uk); dW2_p = delta_prime.get(dk)
                        db1g_p = delta_prime.get(gb); db1u_p = delta_prime.get(ub); db2_p = delta_prime.get(db)

                        if (dW1g_p is not None) and (dW1u_p is not None) and (dW2_p is not None):
                            d_ff = dW1g_p.shape[0]
                            perm, inv_perm = sample_ffn_permutation(
                                d_ff,
                                seed=construct_seed64(t_r, "FFN-P", k, lid),
                                device=dW1g_p.device,
                            )
                            dW1g, dW1u, dW2, db1g, db1u, db2 = inverse_T_on_update_ffn(
                                dW1g_p, dW1u_p, dW2_p, inv_perm,
                                db1_gate_p=db1g_p, db1_up_p=db1u_p, db2_down_p=db2_p
                            )
                            delta_prime[gk], delta_prime[uk], delta_prime[dk] = dW1g, dW1u, dW2
                            if db1g_p is not None: delta_prime[gb] = db1g
                            if db1u_p is not None: delta_prime[ub] = db1u
                            if db2_p  is not None: delta_prime[db] = db2

                # Harmonic accumulate
                w = 1.0 / float(alphas[k - 1])
                S0 += w
                for name, d in delta_prime.items():
                    if name not in S1:
                        S1[name] = d.detach().clone() * w
                    else:
                        S1[name].add_(d, alpha=w)

                del after_sd, delta_prime, pub_sd
                del model_k
                gc.collect()
                if torch.cuda.is_available() and torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
                self._cuda_mem(f"r{r}/k{k}/after_cleanup", device)

            # Finalize bar_delta
            bar_delta = OrderedDict((name, d / S0) for name, d in S1.items())
            delta_norm = state_dict_l2_norm(bar_delta)
            print(f"[Debug] Round {r}: ||Δ̄||₂ = {delta_norm:.6f}")

            for name in theta_prev.keys():
                if torch.is_tensor(theta_prev[name]) and (name in bar_delta) and torch.is_tensor(bar_delta[name]):
                    theta_prev[name] = theta_prev[name] + self.cfg.eta_srv * bar_delta[name]
            self.model.load_state_dict(theta_prev, strict=False)

        if hasattr(self, "state"):
            print(f"[PUM][STEP-CHECK] leaving run(): self.state.global_step={self.state.global_step}", flush=True)

        return self.model

    # Override train(): run server-client PUM protocol instead of HF inner loop
    def train(self, *args, **kwargs):
        """
        Run PUM protocol (R rounds, m copies) instead of HuggingFace Trainer's default training loop.
        This avoids feeding dict keys like 'forget'/'retain' into model.forward().
        """
        # Resolve device to a string understood by .to(...)
        dev = getattr(self, "accelerator", None).device if hasattr(self, "accelerator") else next(self.model.parameters()).device
        if isinstance(dev, torch.device):
            device = f"{dev.type}:{dev.index}" if dev.index is not None else dev.type
        else:
            device = str(dev)

        # Try to use dedicated forget/retain dataloaders if base trainer provides them
        ul_forget = getattr(self, "get_forget_dataloader", None)
        ul_forget = ul_forget() if callable(ul_forget) else None

        ul_retain = getattr(self, "get_retain_dataloader", None)
        ul_retain = ul_retain() if callable(ul_retain) else None

        # Fallback: split from main train dataloader batches {'forget':..., 'retain':...}
        need_retain = str(self.cfg.client_method) in {"GradDiff", "DPO", "NPO", "SimNPO", "SatImp", "UnDIAL", "WGA"}
        if ul_forget is None or (need_retain and ul_retain is None):
            main_dl = self.get_train_dataloader()

            class _SubDataLoader:
                def __init__(self, parent, key):
                    self.parent = parent
                    self.key = key
                def __iter__(self):
                    for batch in iter(self.parent):
                        if self.key == "forget":
                            yield batch["forget"]
                        elif self.key == "retain":
                            yield batch["retain"]
                        elif self.key == "original":
                            yield batch["original"]
                        elif self.key == "alternate":
                            yield batch["alternate"]
                        else:
                            raise KeyError(self.key)

            if ul_forget is None:
                ul_forget = _SubDataLoader(main_dl, "forget")
            if need_retain and ul_retain is None:
                ul_retain = _SubDataLoader(main_dl, "retain")

        # Reference model if required by chosen client_method / retain_loss_type
        ref_model = None
        need_ref = (str(self.cfg.client_method) in {"DPO", "NPO", "UnDIAL"}) or \
                       (getattr(self.cfg, "retain_loss_type", "NLL") == "KL" and str(self.cfg.client_method) in {"GradDiff", "WGA", "SimNPO", "SatImp"})
        if need_ref:
            ref_model = prepare_ref_model(self.model, device=device)

        # Run PUM protocol (base_sd will be loaded inside run() from cfg.base_model_name_or_path)
        # Baseline epochs target comes from trainer args (OpenUnlearning default is 10)
        target_total_epochs = int(getattr(getattr(self, "args", None), "num_train_epochs", 10))

        if self.cfg.client_round_epoch is not None:
            try:
                prod = int(self.cfg.R) * int(self.cfg.client_round_epoch)
            except Exception:
                prod = None
            if prod is not None and prod != target_total_epochs:
                print(
                    f"[PUM][WARN] R({self.cfg.R}) * client_round_epoch({self.cfg.client_round_epoch}) = {prod} "
                    f"!= target_total_epochs({target_total_epochs}). Continue anyway.",
                    flush=True,
                )
                
        self.run(
            ul_forget,
            ul_retain=ul_retain,
            ref_model=ref_model,
            device=device,
            base_sd=None,
        )

        # HF Trainer expects a TrainOutput sometimes, but your pipeline mainly cares that weights are updated.
        return self.model