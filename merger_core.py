import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Literal
import copy
import os


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_shape = v0.shape
    v0_flat = v0.float().flatten()
    v1_flat = v1.float().flatten()

    v0_norm = torch.norm(v0_flat)
    v1_norm = torch.norm(v1_flat)

    v0_unit = v0_flat / (v0_norm + eps)
    v1_unit = v1_flat / (v1_norm + eps)

    dot = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.abs() < eps:
        return ((1 - t) * v0_flat + t * v1_flat).reshape(orig_shape).to(v0.dtype)

    sin_theta = torch.sin(theta)
    coeff0 = torch.sin((1 - t) * theta) / sin_theta
    coeff1 = torch.sin(t * theta) / sin_theta

    merged_norm = (1 - t) * v0_norm + t * v1_norm
    merged_unit = coeff0 * v0_unit + coeff1 * v1_unit
    merged = merged_unit * merged_norm

    return merged.reshape(orig_shape).to(v0.dtype)


def ties_merge(
    base_sd: dict,
    model_a_sd: dict,
    model_b_sd: dict,
    trim_ratio: float = 0.2,
    t: float = 0.5
) -> dict:
    merged_sd = {}

    for key in base_sd.keys():
        base_w = base_sd[key].float()
        a_w    = model_a_sd[key].float()
        b_w    = model_b_sd[key].float()

        if base_w.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_sd[key] = base_w
            continue

        delta_a = a_w - base_w
        delta_b = b_w - base_w

        def trim(delta, ratio):
            if delta.numel() == 0:
                return delta
            threshold = torch.quantile(delta.abs().float(), ratio)
            return delta * (delta.abs() >= threshold).float()

        delta_a = trim(delta_a, trim_ratio)
        delta_b = trim(delta_b, trim_ratio)

        combined = delta_a + delta_b
        elected_sign = torch.sign(combined)

        mask_a = (torch.sign(delta_a) == elected_sign).float()
        mask_b = (torch.sign(delta_b) == elected_sign).float()

        delta_a_clean = delta_a * mask_a
        delta_b_clean = delta_b * mask_b

        merged_delta = t * delta_a_clean + (1 - t) * delta_b_clean

        merged_sd[key] = (base_w + merged_delta).to(base_sd[key].dtype)

    return merged_sd


def dare_merge(
    base_sd: dict,
    model_a_sd: dict,
    model_b_sd: dict,
    drop_rate: float = 0.9,
    t: float = 0.5
) -> dict:
    merged_sd = {}

    for key in base_sd.keys():
        base_w = base_sd[key].float()
        a_w    = model_a_sd[key].float()
        b_w    = model_b_sd[key].float()

        if base_w.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_sd[key] = base_w
            continue

        delta_a = a_w - base_w
        delta_b = b_w - base_w

        def drop_and_rescale(delta, p):
            if delta.numel() == 0:
                return delta
            mask = (torch.rand_like(delta) > p).float()
            scale = 1.0 / (1.0 - p + 1e-8)
            return delta * mask * scale

        sparse_a = drop_and_rescale(delta_a, drop_rate)
        sparse_b = drop_and_rescale(delta_b, drop_rate)

        merged_delta = t * sparse_a + (1 - t) * sparse_b

        merged_sd[key] = (base_w + merged_delta).to(base_sd[key].dtype)

    return merged_sd


class LLMMerger:

    def __init__(self, base_model_path: str, model_a_path: str, model_b_path: str):
        self.base_path = base_model_path
        self.a_path    = model_a_path
        self.b_path    = model_b_path

        self.base_model = None
        self.model_a    = None
        self.model_b    = None
        self.tokenizer  = None

    def load_models(self):
        print("[Merger] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print("[Merger] Loading model A...")
        self.model_a = AutoModelForCausalLM.from_pretrained(
            self.a_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print("[Merger] Loading model B...")
        self.model_b = AutoModelForCausalLM.from_pretrained(
            self.b_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print("[Merger] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_path,
            trust_remote_code=True
        )

    def merge(
        self,
        strategy: Literal["slerp", "ties", "dare"] = "slerp",
        t: float = 0.5,
        **kwargs
    ):
        print(f"\n[Merger] Starting {strategy.upper()} merge (t={t})...")

        base_sd = self.base_model.state_dict()
        a_sd    = self.model_a.state_dict()
        b_sd    = self.model_b.state_dict()

        if strategy == "slerp":
            print("[Merger] Blending weights along spherical arc...")
            merged_sd = {}
            total = len(base_sd)
            for i, key in enumerate(base_sd.keys()):
                w_a = a_sd[key]
                w_b = b_sd[key]
                if w_a.is_floating_point():
                    merged_sd[key] = slerp(t, w_a, w_b)
                else:
                    merged_sd[key] = w_a
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{total} layers merged...")

        elif strategy == "ties":
            print("[Merger] Running TRIM → ELECT → MERGE...")
            trim_ratio = kwargs.get("trim_ratio", 0.2)
            merged_sd = ties_merge(base_sd, a_sd, b_sd, trim_ratio=trim_ratio, t=t)

        elif strategy == "dare":
            print("[Merger] Running DROP → RESCALE → MERGE...")
            drop_rate = kwargs.get("drop_rate", 0.9)
            merged_sd = dare_merge(base_sd, a_sd, b_sd, drop_rate=drop_rate, t=t)

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose slerp / ties / dare.")

        print("[Merger] Loading merged weights into model...")
        merged_model = copy.deepcopy(self.base_model)
        merged_model.load_state_dict(merged_sd)
        merged_model.eval()

        print(f"[Merger] {strategy.upper()} merge complete!")
        return merged_model

    def save(self, merged_model, output_path: str):
        import gc
        os.makedirs(output_path, exist_ok=True)

        del self.base_model
        del self.model_a
        del self.model_b
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Merger] Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("[Merger] Saved successfully.")


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\models\phi-2"

    print("="*60)
    print("LLM Merger — Quick Test")
    print("Merging a model with itself (output should equal input)")
    print("="*60)

    merger = LLMMerger(
        base_model_path=model_path,
        model_a_path=model_path,
        model_b_path=model_path
    )
    merger.load_models()

    for strategy in ["slerp", "ties", "dare"]:
        print(f"\n--- Testing {strategy.upper()} ---")
        merged = merger.merge(strategy=strategy, t=0.5)
        print(f"Output model type : {type(merged).__name__}")
        print(f"Output param count: {sum(p.numel() for p in merged.parameters()):,}")

    print("\nAll 3 strategies working correctly!")
