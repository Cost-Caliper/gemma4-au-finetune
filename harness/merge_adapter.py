#!/usr/bin/env python3
"""Merge the Fireworks LoRA adapter (fw_lora_layout: fused_peft_3d_v1) into
google/gemma-4-26B-A4B-it base weights, shard by shard.

Decode of fused_peft_3d_v1 (from adapter_config.json + base config, all shapes verified):
  - num_experts E=128, rank r=8, alpha=32 -> scale = alpha/r = 4.0
  - standard 2D LoRAs: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj  (delta = B @ A)
  - fused expert LoRAs, block layout (expert e = rows/cols [e*r:(e+1)*r]):
      * "experts.base_layer" = fused gate_up_proj: A [E*r, hidden=2816], B [2*moe_int=1408, E*r]
      * "experts"            = fused down_proj:    A [E*r, moe_int=704], B [hidden=2816, E*r]
  ASSUMPTIONS (validated behaviorally post-merge via trained-item recall):
    1. expert blocks are sequential (not interleaved)
    2. scale applies as alpha/r per expert
Orientation against base 3D tensors is asserted from shapes at runtime.
"""
import json, os, re, sys, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER = sys.argv[1]  # dir containing adapter_model.safetensors
BASE = sys.argv[2]     # HF snapshot dir of google/gemma-4-26B-A4B-it
OUT = sys.argv[3]      # output dir for merged model
_acfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json")))
R, ALPHA, E = _acfg["r"], _acfg["lora_alpha"], 128
SCALE = ALPHA / R
print(f"adapter config: r={R} alpha={ALPHA} scale={SCALE}")

def f32(x):
    return x.to(torch.float32)

# ---- load adapter ----
lora = {}
with safe_open(os.path.join(ADAPTER, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)
print(f"adapter tensors: {len(lora)}")

PREFIX = "base_model.model."
def lkey(layer, mod, ab):
    return f"{PREFIX}model.language_model.layers.{layer}.{mod}.lora_{ab}.weight"

# map: base weight name -> delta computer
def delta_2d(layer, mod):
    A = f32(lora[lkey(layer, mod, "A")]); B = f32(lora[lkey(layer, mod, "B")])
    return (B @ A) * SCALE  # [out, in]

def fused_expert_delta(layer, mod, out_dim, in_dim):
    """Return [E, out, in] delta from fused A [E*r, in], B [out, E*r]."""
    A = f32(lora[lkey(layer, mod, "A")]); B = f32(lora[lkey(layer, mod, "B")])
    assert A.shape == (E * R, in_dim), f"A shape {A.shape} != {(E*R, in_dim)}"
    assert B.shape == (out_dim, E * R), f"B shape {B.shape} != {(out_dim, E*R)}"
    D = torch.zeros((E, out_dim, in_dim), dtype=torch.float32)
    for e in range(E):
        D[e] = (B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :]) * SCALE
    return D

def apply_delta(name, W):
    """Return merged W (same dtype) or None if no delta for this weight."""
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+?)(?:\.weight)?$", name)
    if not m:
        return None
    layer, tail = int(m.group(1)), m.group(2)
    two_d = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    if tail in two_d and lkey(layer, tail, "A") in lora:
        D = delta_2d(layer, tail)
        if W.shape != D.shape and W.shape == D.T.shape:
            D = D.T.contiguous()
        assert W.shape == D.shape, f"{name}: base {W.shape} vs delta {D.shape}"
        return (f32(W) + D).to(W.dtype)
    # fused experts: base 3D tensors
    if "experts" in tail:
        if "gate_up" in tail and lkey(layer, "experts.base_layer", "A") in lora:
            D = fused_expert_delta(layer, "experts.base_layer", out_dim=1408, in_dim=2816)
        elif "down" in tail and lkey(layer, "experts", "A") in lora:
            D = fused_expert_delta(layer, "experts", out_dim=2816, in_dim=704)
        else:
            return None
        if W.shape != D.shape:
            if W.shape == (D.shape[0], D.shape[2], D.shape[1]):
                D = D.permute(0, 2, 1).contiguous()
            else:
                raise AssertionError(f"{name}: base {W.shape} unmatched by delta {D.shape}")
        return (f32(W) + D).to(W.dtype)
    return None

# ---- stream shards ----
os.makedirs(OUT, exist_ok=True)
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
merged_count = 0
for shard in shards:
    out_tensors = {}
    with safe_open(os.path.join(BASE, shard), "pt") as f:
        for name in f.keys():
            W = f.get_tensor(name)
            M = apply_delta(name, W)
            if M is not None:
                merged_count += 1
                out_tensors[name] = M
            else:
                out_tensors[name] = W
    save_file(out_tensors, os.path.join(OUT, shard), metadata={"format": "pt"})
    print(f"wrote {shard} ({len(out_tensors)} tensors)")

# copy configs/tokenizer/index
for fn in os.listdir(BASE):
    if fn.endswith((".json", ".model", ".txt")) and not fn.startswith("model-"):
        shutil.copy(os.path.join(BASE, fn), os.path.join(OUT, fn))
print(f"DONE: merged deltas into {merged_count} weight tensors -> {OUT}")
expected = 30 * 9  # 30 layers x (4 attn + 3 mlp + 2 fused expert groups)
print(f"expected ~{expected}, got {merged_count}")
