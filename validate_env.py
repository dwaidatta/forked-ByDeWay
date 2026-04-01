
import sys
import traceback

print("Starting validation...", flush=True)

def test(name, func):
    print(f"Testing {name}...", end="", flush=True)
    try:
        func()
        print(" OK", flush=True)
    except Exception:
        print(" FAILED", flush=True)
        traceback.print_exc()

def check_torch():
    import torch
    print(f" {torch.__version__}", end="")
    x = torch.tensor([1.0])
    if torch.cuda.is_available():
        print(" (CUDA)", end="")
    else:
        print(" (CPU)", end="")

def check_cv2():
    import cv2
    print(f" {cv2.__version__}", end="")

def check_pil():
    from PIL import Image
    pass

def check_sentencepiece():
    import sentencepiece
    print(f" {sentencepiece.__version__}", end="")

def check_transformers():
    import transformers
    print(f" {transformers.__version__}", end="")
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

if __name__ == "__main__":
    test("Torch", check_torch)
    test("OpenCV", check_cv2)
    test("Pillow", check_pil)
    test("SentencePiece", check_sentencepiece)
    # Transformers last as it downloads
    test("Transformers (download)", check_transformers)
    print("Validation done.")
