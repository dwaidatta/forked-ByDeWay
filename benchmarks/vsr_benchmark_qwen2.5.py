"""
VSR (Visual Spatial Reasoning) Benchmark — Qwen 2.5-VL
=======================================================
Evaluates Qwen 2.5-VL on the VSR dataset.
"""

import os
import sys
import json
import re
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests as http_requests
from io import BytesIO

# Hugging Face & Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner
from src.depth_captioning.spatial_analysis import RELATION_TO_CATEGORY

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

def parse_vsr_caption(caption, relation):
    text = caption.strip().rstrip(".")
    for marker in [f" is {relation} the ", f" {relation} the "]:
        if marker in text:
            left, right = text.split(marker, 1)
            subj = left[4:] if left.startswith("The ") else left
            return subj.strip(), right.strip()
    return None, None

def normalize_true_false(text: str) -> str:
    if not text: return "unknown"
    t = _NON_ALNUM_RE.sub(" ", text.strip().lower()).strip()
    first = t.split(" ", 1)[0] if t else ""
    if first in ("yes", "y", "true", "1", "correct"): return "true"
    if first in ("no", "n", "false", "0", "incorrect", "wrong"): return "false"
    return "unknown"

def build_ldp_context(depth_captioner, image, mode):
    layer_names = ["Closest", "Farthest", "Mid Range"]
    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
        image, top_threshold=70, bottom_threshold=30, return_masks=True
    )
    spatial = depth_captioner.spatial_analyzer.analyze(np.array(image), masks, max_relations_per_layer=6)
    blocks = []
    for idx, layer_np in enumerate(layer_imgs):
        caption = depth_captioner.captioner.get_caption(layer_np)
        block = f"{layer_names[idx]}: {caption}"
        if mode == "ldp_spatial" and spatial[idx]:
            block += f"\nSpatial Relationships: {spatial[idx]}"
        blocks.append(block)
    return "\n----\n".join(blocks)

def build_vsr_spatial_context(depth_captioner, image):
    image_array = np.array(image)
    try:
        depth_map = depth_captioner.depth_context.predict_depth(image_array[:, :, ::-1])
    except Exception: depth_map = None
    return depth_captioner.spatial_analyzer.analyze_vsr_for_caption(image_array, depth_map=depth_map)

def ask_qwen_true_false(model, processor, image, caption, context, vsr_spatial, mode, max_new_tokens):
    if mode == "baseline":
        prompt = f"Look at the image carefully. Is the statement '{caption}' true or false? Answer only one word: true or false."
    elif mode == "ldp":
        prompt = (
            "You are evaluating if a spatial description is true or false.\n"
            "Use the image as primary evidence and the depth context as hints.\n"
            f"Depth Context:\n{context}\n\n"
            f"Statement: \"{caption}\"\n"
            "Answer only 'true' or 'false'."
        )
    else:  # ldp_spatial
        prompt = (
            "Verify if the following statement is 'true' or 'false'.\n"
            "Use the 'Depth Context' and 'Spatial Advisory Data' to resolve ambiguity.\n"
            "Answer ONLY 'true' or 'false'.\n\n"
            f"Depth Context:\n{context}\n"
            f"Spatial Advisory Data:\n{vsr_spatial}\n"
            f"Statement to Verify: \"{caption}\""
        )

    messages = [{"role": "user", "content": [{"type": "image", "image": image, "max_pixels": 313600}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return normalize_true_false(raw), raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mode", type=str, default="ldp_spatial", choices=["baseline", "ldp", "ldp_spatial"])
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"\n[1/4] Initializing Pipeline...")
    depth_captioner = None
    if args.mode != "baseline":
        depth_captioner = DepthBlipCaptioner(device=torch.device(args.device))

    print(f"\n[2/4] Loading Qwen Model...")
    # EXACT ORIGINAL LOADING CODE
    _has_cuda = torch.cuda.is_available()
    _device_map = "auto" if _has_cuda else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        dtype=torch.bfloat16,
        device_map=_device_map,
        attn_implementation="sdpa",
    )
    qwen_model.eval()
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path)

    print(f"\n[3/4] Loading Dataset...")
    ds = load_dataset("cambridgeltl/vsr_random", split="test")
    if args.dry_run: ds = ds.select(range(min(5, len(ds))))

    print(f"\n[4/4] Evaluating...")
    out_path = f"data/vsr_qwen_{args.mode}_predictions.jsonl"
    
    y_true, y_pred = [], []
    per_category = defaultdict(lambda: {"y_true": [], "y_pred": []})
    os.makedirs("data", exist_ok=True)
    
    # Category mapping
    from src.depth_captioning.spatial_analysis import RELATION_TO_CATEGORY

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            try:
                caption, label, relation, link = row["caption"], row["label"], row["relation"], row["image_link"]
                category = RELATION_TO_CATEGORY.get(relation, "unallocated")
                resp = http_requests.get(link, timeout=15)
                image = Image.open(BytesIO(resp.content)).convert("RGB")
                
                context, vsr_spatial = "", ""
                if depth_captioner:
                    subj, obj = parse_vsr_caption(caption, relation)
                    if subj and obj: depth_captioner.spatial_analyzer.set_classes([subj, obj])
                    context = build_ldp_context(depth_captioner, image, args.mode)
                    if args.mode == "ldp_spatial":
                        vsr_spatial = build_vsr_spatial_context(depth_captioner, image)
                
                norm, raw = ask_qwen_true_false(qwen_model, qwen_processor, image, caption, context, vsr_spatial, args.mode, 10)
                pred_label = 1 if norm == "true" else 0
                
                y_true.append(label)
                y_pred.append(pred_label)
                per_category[category]["y_true"].append(label)
                per_category[category]["y_pred"].append(pred_label)
                
                f_out.write(json.dumps({"idx": idx, "gt": label, "pred": pred_label, "cat": category, "raw": raw}) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"Error: {e}")

    # Results Breakdown
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  VSR Results (Qwen) — Mode: {args.mode}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*60}")

    print(f"\nPer-Category Breakdown:")
    print(f"{'Category':<16} {'N':>5} {'Acc':>7} {'F1':>7}")
    for cat in sorted(per_category.keys()):
        yt, yp = per_category[cat]["y_true"], per_category[cat]["y_pred"]
        if not yt: continue
        c_acc = accuracy_score(yt, yp)
        c_f1 = f1_score(yt, yp, zero_division=0)
        print(f"{cat:<16} {len(yt):>5} {c_acc:>7.4f} {c_f1:>7.4f}")

if __name__ == "__main__":
    main()
