import requests
from PIL import Image
from src.depth_captioning.depth_blip import DepthBlipCaptioner
import sys
import traceback

def main():
    print("Starting quick_start.py...", flush=True)
    try:
        print("Importing modules...", flush=True)
        # Verify imports again just in case
        import requests
        
        print("Initializing DepthBlipCaptioner (using BLIP model)...", flush=True)
        # Initialize the depth-aware captioning pipeline with BLIP
        depth_captioner = DepthBlipCaptioner() 
        print("DepthBlipCaptioner initialized.", flush=True)
        
        # Load the image from a URL
        url = "https://c8.alamy.com/comp/P6YB78/los-angeles-usa-june-29-unidentified-random-people-in-the-streets-of-downtown-of-los-angeles-ca-on-june-29-2018-P6YB78.jpg"
        print(f"Downloading image from {url}...", flush=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        print("Image downloaded and opened.", flush=True)
        
        # Generate and print the depth-based structured caption
        print("Generating caption (this may take a while as models download)...", flush=True)
        full_caption_string = depth_captioner.get_caption_with_depth(image)
        print("\nGenerated Caption:")
        print(full_caption_string)
        print("\nSuccess! The project is running fine.")
        
    except Exception:
        print("\nError encountered:", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
