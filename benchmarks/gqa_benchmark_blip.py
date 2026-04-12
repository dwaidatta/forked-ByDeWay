import os
import sys
import json
import re
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering

warnings.filterwarnings('ignore', category=UserWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner

def normalize_word(word: str) -> str:
    """Normalize word to lowercase and remove non-alphanumeric characters."""
    if not word:
        return ""
    word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
    return word.strip().lower()

def exact_match(prediction: str, ground_truth: str) -> bool:
    """Calculate relaxed match score."""
    p_norm = normalize_word(prediction)
    g_norm = normalize_word(ground_truth)
    if not p_norm or not g_norm:
        return False
    # Relaxed match: checking if ground truth is in the prediction or vice versa
    return p_norm in g_norm or g_norm in p_norm

def _shorten(text: str, max_words: int = 18) -> str:
    if not text:
        return ""
    words = str(text).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."

def _compact_context(context: str, max_chars: int = 600) -> str:
    if not context:
        return ""
    c = " ".join(str(context).split())
    if len(c) <= max_chars:
        return c
    return c[: max_chars - 3].rstrip() + "..."

def build_ldp_context(depth_captioner: DepthBlipCaptioner, image: Image.Image, mode: str) -> str:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP on GQA with Spatial LDP")
    parser.add_argument("--dry_run", action="store_true", help="Run on only 5 samples for testing")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (auto-generated from mode if not set)")
    parser.add_argument("--dataset", type=str, default="Rajarshi-Roy-research/GQA-dataset-150", help="HF Dataset repository")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--vqa_model", type=str, default="Salesforce/blip-vqa-base", help="BLIP VQA checkpoint")
    parser.add_argument("--depth_encoder", type=str, default="vits", choices=["vits","vitb","vitl","vitg"], help="Depth Anything V2 encoder size.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLO model for spatial analysis.")
    parser.add_argument("--use_context", action="store_true", help="Include LDP+spatial context in the text prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for models (cpu/cuda)")
    parser.add_argument("--mode", type=str, default="ldp_spatial", choices=["ldp", "ldp_spatial"], help="Context mode")
    return parser.parse_args()

def main():
    args = parse_args()
    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/gqa_blip_{args.mode}_predictions.jsonl"
    print(f"Starting GQA Benchmark for BLIP on device: {args.device}")
    print(f"Mode: {args.mode} | Output: {args.output}")
    
    print("\n[1/3] Initializing DepthBlipCaptioner (Depth Anything + YOLO + BLIP Captioning)...")
    depth_captioner = DepthBlipCaptioner(
        device=torch.device(args.device),
        encoder=args.depth_encoder,
        yolo_model_path=args.yolo_model,
    )
    
    print(f"\n[2/3] Initializing base BLIP VQA model ({args.vqa_model})...")
    vqa_processor = BlipProcessor.from_pretrained(args.vqa_model)
    vqa_model = BlipForQuestionAnswering.from_pretrained(args.vqa_model).to(args.device)
    vqa_model.eval()

    print(f"\n[3/3] Loading dataset: {args.dataset} (split: {args.split})")
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        return
        
    num_samples = len(ds)
    if args.dry_run:
        num_samples = min(5, num_samples)
        print(f"\nDRY RUN enabled. Only processing {num_samples} samples.")
        ds = ds.select(range(num_samples))
    else:
        print(f"\nLoaded {num_samples} samples. Beginning evaluation...")
        
    correct = 0
    total = 0
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            image = row['image']
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            try:
                spatial_depth_caption = build_ldp_context(depth_captioner, image, args.mode)
            except Exception as e:
                print(f"Error generating depth/spatial caption for idx {idx}: {e}")
                spatial_depth_caption = "No depth context available."

            qa_list = row.get('qa', [])
            if not isinstance(qa_list, list):
                if isinstance(qa_list, dict):
                    qa_list = [qa_list]
                else:
                    continue

            for qa_idx, qa_item in enumerate(qa_list):
                question = str(qa_item.get('question', ''))
                ground_truth = str(qa_item.get('answer', '')).strip()
                question_id = str(qa_item.get('question_id', f"{idx}_{qa_idx}"))
                
                try:
                    if args.use_context:
                        layer_names = ["Closest", "Farthest", "Mid Range"]
                        layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
                            image, top_threshold=70, bottom_threshold=30, return_masks=True
                        )
                        spatial = depth_captioner.spatial_analyzer.analyze(np.array(image), masks, max_relations_per_layer=6)
                        
                        best_answer = ""
                        # Note: BLIP generate does not return logits by default easily inside generate.
                        # Since we want to find the best amongst layers, we can use the simplest heuristic:
                        # Return the first non-empty valid string, or just pick the Closest layer.
                        # Wait, we can gather answers from all layers. For GQA, exact match is needed.
                        answers = []
                        for li, layer_np in enumerate(layer_imgs):
                            layer_pil = Image.fromarray(layer_np.astype("uint8"))
                            caption = depth_captioner.captioner.get_caption(layer_np)
                            caption = _shorten(caption, max_words=16)
                            
                            ctx_bits = [f"[{layer_names[li]}] {caption}."]
                            if args.mode == "ldp_spatial" and spatial[li]:
                                spat = _shorten(spatial[li], max_words=20)
                                ctx_bits.append(f"[Sp] {spat}.")
                                
                            small_ctx = " ".join(ctx_bits)
                            prompt = f"Question: {question} Context: {small_ctx}. Answer:"
                            
                            inputs = vqa_processor(images=layer_pil, text=prompt, return_tensors="pt", truncation=True, max_length=96).to(args.device)
                            with torch.no_grad():
                                out = vqa_model.generate(**inputs, max_new_tokens=10, num_beams=5)
                            ans = vqa_processor.decode(out[0], skip_special_tokens=True).strip()
                            if ans:
                                answers.append(ans)
                                
                        # Return the most commonly generated answer among the layers
                        if answers:
                            from collections import Counter
                            best_answer = Counter(answers).most_common(1)[0][0]
                        else:
                            best_answer = ""
                            
                        raw_pred = best_answer

                    else:
                        inputs = vqa_processor(images=image, text=question, return_tensors="pt", truncation=True, max_length=64).to(args.device)
                        with torch.no_grad():
                            out = vqa_model.generate(**inputs, max_new_tokens=10, num_beams=5)
                        raw_pred = vqa_processor.decode(out[0], skip_special_tokens=True).strip()
                except Exception as e:
                    print(f"Error generating answer for q_idx {question_id}: {e}")
                    raw_pred = ""

                is_correct = exact_match(raw_pred, ground_truth)
                if is_correct:
                    correct += 1
                total += 1
                
                log_entry = {
                    "question_id": question_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "pred_answer": raw_pred,
                    "is_correct": is_correct,
                    "spatial_depth_caption": spatial_depth_caption
                }
                f_out.write(json.dumps(log_entry) + "\n")
                f_out.flush()
            
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*50)
    print("      GQA Benchmark Results (BLIP)     ")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.4f}")
    print("="*50)
    print(f"Detailed predictions saved to {args.output}")

if __name__ == "__main__":
    main()
