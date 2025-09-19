# generate.py
"""
Generate MNIST samples from a trained MeanFlow model.
Uses EMA weights if available.
"""

import os
import torch
from torchvision import utils
from meanflow_mnist_optimized import (
    UNetSmall, FourierTimeEmbed, unnormalize_img,
    sample_and_save, CHECKPOINT_FILE, DEVICE
)

# ---- Config ----
N_SAMPLES = int(os.environ.get("N_SAMPLES", 64))
SAMPLE_STEPS = int(os.environ.get("SAMPLE_STEPS", 200))  # can increase for sharper results
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "generated_digits.png")

def load_model_and_temembed(checkpoint_file=CHECKPOINT_FILE):
    model = UNetSmall(time_dim=128, base_ch=128).to(DEVICE)
    tembed = FourierTimeEmbed(dim=128, scale=50.0).to(DEVICE)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"❌ No checkpoint found at {checkpoint_file}")

    ck = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(ck["ema_state"] or ck["model_state"])
    tembed.load_state_dict(ck["tembed_state"])
    model.eval()
    print(f"✅ Loaded model from {checkpoint_file}")
    return model, tembed

def generate_samples():
    model, tembed = load_model_and_temembed()
    imgs = sample_and_save(
        model, tembed, step=0,
        n_samples=N_SAMPLES, steps=SAMPLE_STEPS,
        folder="outputs", filename=OUTPUT_FILE, device=DEVICE
    )
    print(f"✅ Generated {N_SAMPLES} samples -> outputs/{OUTPUT_FILE}")

if __name__ == "__main__":
    with torch.no_grad():
        generate_samples()
