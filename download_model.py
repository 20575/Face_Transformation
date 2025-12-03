# download_model.py - Download inswapper_128.onnx from stable Hugging Face repo (ezioruan, 2025 verified)
import os
from huggingface_hub import hf_hub_download

print("🚀 Downloading inswapper_128.onnx pretrained model (~554 MB, 1 time only)...")

# InsightFace expected path
insightface_path = os.path.expanduser("~/.insightface/models")
os.makedirs(insightface_path, exist_ok=True)
model_filename = "inswapper_128.onnx"

# Download from stable repo
model_path = hf_hub_download(
    repo_id="ezioruan/inswapper_128.onnx",
    filename=model_filename,
    local_dir=insightface_path,
    local_dir_use_symlinks=False
)

print(f"✅ Model downloaded successfully to: {model_path}")
print("Now you can run the swap script – InsightFace will detect it automatically!")