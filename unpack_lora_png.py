"""Decode LoRA PNG(s) written by TrainGluePackLora back to a .safetensors file.

    python unpack_lora_png.py out_dir chunk1.png [chunk2.png ...]
"""
import os
import sys

import numpy as np
from PIL import Image

from m3ds import unpack

if __name__ == "__main__":
    out_dir, pngs = sys.argv[1], sys.argv[2:]
    Image.MAX_IMAGE_PIXELS = None
    arrays = [np.array(Image.open(p).convert("RGB")) for p in pngs]
    data, name = unpack(arrays)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(name))
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {path} ({len(data)} bytes)")
