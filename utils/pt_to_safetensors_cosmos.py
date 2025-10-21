#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch import Tensor
from safetensors.torch import load_file as st_load_file, save_file as st_save_file


def load_reference(ref_path: Path) -> Dict[str, Tensor]:
    ref = st_load_file(str(ref_path), device="cpu")
    return {k: v for k, v in ref.items()}


def is_tensor_dict(d) -> bool:
    return isinstance(d, dict) and all(isinstance(v, torch.Tensor) for v in d.values())


def extract_state_dict(obj) -> Dict[str, Tensor]:
    if is_tensor_dict(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("state_dict", "ema_state_dict", "model", "module", "net", "generator"):
            if k in obj and is_tensor_dict(obj[k]):
                return obj[k]
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        sd = obj.state_dict()
        if is_tensor_dict(sd):
            return sd
    raise ValueError("Could not find a flat {str: Tensor} state_dict in the .pt file.")


# ---------------- Block mapping (net.blocks.i.*  -> transformer_blocks.i.*) ----------------

_re_block = re.compile(r"^(?:[^.]+\.)?(?:net\.)?blocks\.(\d+)\.(.+)$")

def map_block_key(k_tail: str) -> Tuple[str, bool]:
    m = re.match(r"^self_attn\.(q|k|v)_proj\.(weight|bias)$", k_tail)
    if m:
        which, wb = m.groups()
        return f"attn1.to_{which}.{wb}", True
    m = re.match(r"^self_attn\.output_proj\.(weight|bias)$", k_tail)
    if m:
        (wb,) = m.groups()
        return f"attn1.to_out.0.{wb}", True
    m = re.match(r"^self_attn\.(q|k)_norm\.(weight|bias)$", k_tail)
    if m:
        which, wb = m.groups()
        return f"attn1.norm_{which}.{wb}", True

    m = re.match(r"^cross_attn\.(q|k|v)_proj\.(weight|bias)$", k_tail)
    if m:
        which, wb = m.groups()
        return f"attn2.to_{which}.{wb}", True
    m = re.match(r"^cross_attn\.output_proj\.(weight|bias)$", k_tail)
    if m:
        (wb,) = m.groups()
        return f"attn2.to_out.0.{wb}", True
    m = re.match(r"^cross_attn\.(q|k)_norm\.(weight|bias)$", k_tail)
    if m:
        which, wb = m.groups()
        return f"attn2.norm_{which}.{wb}", True

    m = re.match(r"^mlp\.layer1\.(weight|bias)$", k_tail)
    if m:
        (wb,) = m.groups()
        return f"ff.net.0.proj.{wb}", True
    m = re.match(r"^mlp\.layer2\.(weight|bias)$", k_tail)
    if m:
        (wb,) = m.groups()
        return f"ff.net.2.{wb}", True

    m = re.match(r"^adaln_modulation_(self_attn|cross_attn|mlp)\.(1|2)\.(weight|bias)$", k_tail)
    if m:
        which, lin, wb = m.groups()
        norm_id = {"self_attn": "norm1", "cross_attn": "norm2", "mlp": "norm3"}[which]
        return f"{norm_id}.linear_{lin}.{wb}", True

    return k_tail, False


# ---------------- Top-level mapping (more aliases) ----------------

TOP_LEVEL_PATTERNS = [
    # patch_embed.proj.{weight,bias}
    (re.compile(r"^(?:[^.]+\.)*(?:net\.)?(?:video_)?(?:spatial_)?(?:patch_embed|embed_patch|patchify)\.proj\.(weight|bias)$"),
     lambda m: f"patch_embed.proj.{m.group(1)}"),
    # proj_out.{weight,bias}
    (re.compile(r"^(?:[^.]+\.)*(?:net\.)?(?:proj_out|out_proj|final_proj|to_out|to_logits)\.(weight|bias)$"),
     lambda m: f"proj_out.{m.group(1)}"),
    # time_embed.norm.{weight,bias}
    (re.compile(r"^(?:[^.]+\.)*(?:net\.)?(?:time_embed|time_mlp|t_embed|temb|timestep_embedder)\.norm\.(weight|bias)$"),
     lambda m: f"time_embed.norm.{m.group(1)}"),
    # time_embed.t_embedder.linear_{1,2}.{weight,bias}
    (re.compile(r"^(?:[^.]+\.)*(?:net\.)?(?:time_embed|time_mlp|t_embed|temb|timestep_embedder)\.(?:t_embedder|mlp)\.linear_(1|2)\.(weight|bias)$"),
     lambda m: f"time_embed.t_embedder.linear_{m.group(1)}.{m.group(2)}"),
    # norm_out.linear_{1,2}.{weight,bias}
    (re.compile(r"^(?:[^.]+\.)*(?:net\.)?(?:norm_out|out_norm|final_norm|post_norm|ln_out)\.linear_(1|2)\.(weight|bias)$"),
     lambda m: f"norm_out.linear_{m.group(1)}.{m.group(2)}"),
]

def rename_key(k: str) -> Tuple[str, bool]:
    m = _re_block.match(k)
    if m:
        idx, tail = m.groups()
        mapped_tail, ok = map_block_key(tail)
        if ok:
            return f"transformer_blocks.{idx}.{mapped_tail}", True

    # do not let top-level rules accidentally grab block params
    if ".blocks." in k:
        return k, False

    for pat, builder in TOP_LEVEL_PATTERNS:
        m = pat.match(k)
        if m:
            return builder(m), True

    return k, False


# ---------------- Verification and helpers ----------------

def ensure_shapes_and_dtypes(mapped: Dict[str, Tensor], ref: Dict[str, Tensor]) -> Tuple[bool, str, Dict[str, Tensor]]:
    problems = []
    for k in ref.keys():
        if k not in mapped:
            problems.append(f"missing key: {k}")
            continue
        if tuple(mapped[k].shape) != tuple(ref[k].shape):
            problems.append(f"shape mismatch at {k}: src {tuple(mapped[k].shape)} vs ref {tuple(ref[k].shape)}")
        if mapped[k].dtype != ref[k].dtype:
            try:
                mapped[k] = mapped[k].to(dtype=ref[k].dtype)
            except Exception as e:
                problems.append(f"dtype cast failed at {k}: {mapped[k].dtype} -> {ref[k].dtype} ({e})")
    ok = len(problems) == 0
    return ok, "\n".join(problems), mapped


def keyword_score(name: str, tokens: List[str]) -> int:
    name_l = name.lower()
    return sum(1 for t in tokens if t in name_l)


def shape_driven_guess(missing: List[str], unmapped: Dict[str, Tensor], ref: Dict[str, Tensor]) -> Dict[str, Tensor]:
    target_tokens = {
        "patch_embed.proj.weight": ["patch", "embed", "proj", "weight"],
        "proj_out.weight": ["proj", "out", "weight"],
        "time_embed.norm.weight": ["time", "temb", "norm", "weight"],
        "time_embed.t_embedder.linear_1.weight": ["time", "temb", "t_embedder", "linear_1", "weight"],
        "time_embed.t_embedder.linear_2.weight": ["time", "temb", "t_embedder", "linear_2", "weight"],
        "norm_out.linear_1.weight": ["norm", "out", "linear_1", "weight"],
        "norm_out.linear_2.weight": ["norm", "out", "linear_2", "weight"],
    }
    added: Dict[str, Tensor] = {}

    for tgt in missing:
        if tgt not in ref:
            continue
        tgt_shape = tuple(ref[tgt].shape)
        cands = [(k, v) for k, v in unmapped.items() if isinstance(v, torch.Tensor) and tuple(v.shape) == tgt_shape]
        if not cands:
            print(f"[shape-miss] {tgt}: no same-shape candidate")
            continue
        # rank by keyword overlap
        toks = target_tokens.get(tgt, [])
        ranked = sorted(cands, key=lambda kv: (-keyword_score(kv[0], toks), kv[0]))
        best = ranked[0]
        # accept only if it is unique and reasonably scored
        if len(ranked) == 1 or keyword_score(best[0], toks) > keyword_score(ranked[1][0], toks):
            print(f"[shape-map] {tgt} <= {best[0]}  shape={tgt_shape}")
            added[tgt] = best[1]
        else:
            print(f"[shape-ambiguous] {tgt}: top two {ranked[0][0]} , {ranked[1][0]} — skipping")
    return added


def main():
    ap = argparse.ArgumentParser(description="Convert Cosmos-Predict2 training .pt → Diffusers-style .safetensors.")
    ap.add_argument("--src_pt", type=Path, required=True)
    ap.add_argument("--ref_safetensors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no_shape_guess", action="store_true", help="Disable shape-driven fallback mapping.")
    args = ap.parse_args()

    ref = load_reference(args.ref_safetensors)
    ref_keys = set(ref.keys())
    print(f"[info] reference tensors: {len(ref_keys)}")

    print(f"[info] loading source: {args.src_pt}")
    obj = torch.load(str(args.src_pt), map_location="cpu", weights_only=True)
    src_sd = extract_state_dict(obj)
    # tensors only
    src_sd = {k: v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    print(f"[info] source tensors after filtering: {len(src_sd)}")

    mapped: Dict[str, Tensor] = {}
    unmapped: Dict[str, Tensor] = {}
    for k, v in src_sd.items():
        if k.endswith("attn_op._extra_state"):
            continue
        new_k, ok = rename_key(k)
        if ok:
            mapped[new_k] = v
        else:
            unmapped[k] = v

    # keep only keys present in reference
    mapped = {k: v for k, v in mapped.items() if k in ref_keys}
    print(f"[info] mapped → {len(mapped)} tensors; unmapped → {len(unmapped)} tensors")

    missing = sorted(ref_keys - set(mapped.keys()))
    extra = sorted(set(mapped.keys()) - ref_keys)
    print(f"[check] still missing vs reference: {len(missing)}")
    print(f"[check] extras vs reference (should be 0): {len(extra)}")

    # shape-driven fallback for the 7 common top-level weights
    top7 = {
        "norm_out.linear_1.weight",
        "norm_out.linear_2.weight",
        "patch_embed.proj.weight",
        "proj_out.weight",
        "time_embed.norm.weight",
        "time_embed.t_embedder.linear_1.weight",
        "time_embed.t_embedder.linear_2.weight",
    }
    need = [k for k in missing if k in top7]
    if need and not args.no_shape_guess:
        guessed = shape_driven_guess(need, unmapped, ref)
        mapped.update(guessed)
        missing = sorted(ref_keys - set(mapped.keys()))
        print(f"[after-shape-guess] still missing: {len(missing)}")

    ok, report, mapped = ensure_shapes_and_dtypes(mapped, ref)
    if not ok:
        print("[error] shape/dtype issues:\n" + report)
        # show first 40 unmapped names to help refinement
        to_show = [k for k in sorted(unmapped.keys()) if any(t in k.lower() for t in ["patch", "proj", "time", "temb", "embed", "norm", "out"])]
        if to_show:
            print("[hint] unmapped candidates:")
            for k in to_show[:40]:
                print("   ", k)
        sys.exit(1)
    else:
        print("[info] shapes and dtypes match the reference.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    st_save_file(mapped, str(args.out))
    print(f"[info] wrote: {args.out}")

    saved = st_load_file(str(args.out), device="cpu")
    if set(saved.keys()) != ref_keys:
        diff1 = sorted(ref_keys - set(saved.keys()))
        diff2 = sorted(set(saved.keys()) - ref_keys)
        print("[error] post-save keyset mismatch.")
        if diff1:
            print(f"  missing after save ({len(diff1)}): {diff1[:20]}{' ...' if len(diff1)>20 else ''}")
        if diff2:
            print(f"  unexpected after save ({len(diff2)}): {diff2[:20]}{' ...' if len(diff2)>20 else ''}")
        sys.exit(1)

    print("[done] conversion succeeded and verified.")


if __name__ == "__main__":
    main()
