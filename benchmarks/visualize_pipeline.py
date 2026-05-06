import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.depth_captioning.depth_kosmos import DepthContextCreator
from src.depth_captioning.spatial_analysis import SpatialAnalyzer

from src.depth_captioning.depth_blip import BlipCaptioner

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def main():
    # Load sample image (VSR dataset example: laptop facing sandwich / office scene)
    img_url = "http://images.cocodataset.org/train2017/000000519404.jpg"
    print(f"Downloading sample image from {img_url}...")
    pil_img = download_image(img_url)
    img_array = np.array(pil_img)

    print("Initializing modules (this may take a moment to load weights)...")
    depth_creator = DepthContextCreator(encoder="vits")
    spatial_analyzer = SpatialAnalyzer(model_path="yolov8n.pt")
    captioner = BlipCaptioner(ckpt="Salesforce/blip-image-captioning-base")

    print("Running Depth Estimation...")
    depth_map = depth_creator.predict_depth(img_array[:, :, ::-1])
    
    print("Running Depth Segmentation...")
    segmented_images, masks = depth_creator.make_depth_context_img(
        pil_img, top_threshold=70, bottom_threshold=30, return_masks=True
    )
    
    print("Running Object Detection...")
    objects = spatial_analyzer._detect_objects(img_array)
    
    print("Extracting Spatial Relations...")
    spatial_relations_text = spatial_analyzer.analyze_vsr_for_caption(img_array, depth_map=depth_map)

    print("Generating Depth Captions dynamically using BLIP...")
    labels = ["Closest", "Farthest", "Mid-Range"]
    depth_captions = []
    # Note: DepthContextCreator outputs: [top30 (closest), bottom30 (farthest), mid40 (mid)]
    for i in range(3):
        cap = captioner.get_caption(segmented_images[i])
        depth_captions.append(f"{labels[i]}: {cap}")

    # Create the figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ByDeWay-V2 Pipeline Visualization", fontsize=20, fontweight='bold', y=0.98)

    # 1. Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(pil_img)
    ax1.set_title("Input Image", fontsize=14)
    ax1.axis('off')

    # 2. Depth Map
    ax2 = plt.subplot(2, 3, 2)
    # Depth map is typically 1 channel, we colorize it using inferno
    ax2.imshow(depth_map, cmap='inferno')
    ax2.set_title("Depth Map", fontsize=14)
    ax2.axis('off')

    # 3. Bounding Boxes
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(pil_img)
    ax3.set_title("Detected Objects (Bounding Boxes)", fontsize=14)
    ax3.axis('off')
    
    # Draw boxes
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        w = x2 - x1
        h = y2 - y1
        label = obj.get("display_label", obj["label"])
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='#00ff00', facecolor='none')
        ax3.add_patch(rect)
        ax3.text(x1, y1 - 5, label, color='white', fontsize=10, backgroundcolor='#00ff00', weight='bold')

    # 4. Text Prompt Area (Spans the bottom row)
    ax_text = plt.subplot(2, 1, 2)
    ax_text.axis('off')
    
    prompt_text = "== Final Prompt sent to MLLM ==\n\n"
    prompt_text += "[Depth-Based Layer Descriptions]\n"
    prompt_text += "\n".join(depth_captions) + "\n\n"
    
    prompt_text += "[Explicit Spatial Relations]\n"
    
    # Split spatial relations for better formatting
    if spatial_relations_text:
        rels = spatial_relations_text.split(". ")
        for r in rels:
            if r:
                prompt_text += f"- {r}.\n"
    else:
        prompt_text += "- No distinct spatial relations found.\n"
        
    prompt_text += "\n[Query]\nIs the laptop facing the sandwich?"

    # Draw a styled text box
    props = dict(boxstyle='round,pad=1', facecolor='#e8f4f8', alpha=0.8, edgecolor='#3498db', linewidth=2)
    ax_text.text(0.5, 0.5, prompt_text, transform=ax_text.transAxes, fontsize=14,
                 verticalalignment='center', horizontalalignment='center', bbox=props, family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_dir = r"D:/ByDeWay/ByDeWay-Depth-Captioning/figures"
    out_path = os.path.join(out_dir, "pipeline_visualization.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {out_path}")

if __name__ == "__main__":
    main()
