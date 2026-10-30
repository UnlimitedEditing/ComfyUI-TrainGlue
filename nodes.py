import os
import time

import folder_paths
import comfy.sd
import comfy.utils
from comfy.cli_args import args


class TrainGlueUNETLoader:
    """UNETLoader that loads with comfy-aimdo DynamicVRAM disabled for this model.

    Core TrainLoraNode crashes on Graydient (comfy-aimdo 0.6.0) when the model is a dynamic-VRAM patcher:
    'aimdo memory compile error' with gradient checkpointing, a hard 'CUDA error: invalid argument' without.
    comfy.sd.load_diffusion_model(disable_dynamic=True) gives a plain ModelPatcher that loads fully to the GPU.
    That alone is not enough: aimdo's 'model compiler' malloc graph (comfy.model_prefetch) records allocations per
    forward and raises 'aimdo memory compile error' when checkpointing re-runs a block during backward. It is gated
    by args.disable_comfy_compiler, so this loader also switches it off for the rest of the job.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "model/training"

    def load(self, unet_name, weight_dtype):
        import torch
        opts = {}
        if weight_dtype == "fp8_e4m3fn":
            opts["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            opts["dtype"] = torch.float8_e4m3fn
            opts["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            opts["dtype"] = torch.float8_e5m2
        args.disable_comfy_compiler = True
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        return (comfy.sd.load_diffusion_model(path, model_options=opts, disable_dynamic=True),)


class TrainGluePackLora:
    """Pack a trained LoRA (LORA_MODEL) into lossless M3DS PNG(s) so it comes back through an image output.

    Graydient only collects image outputs from a job; the .safetensors SaveLoRA writes is not delivered.
    Decode with unpack_lora_png.py (this repo) or Meshsmuggler's unsmuggle.html.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora": ("LORA_MODEL",),
            "filename_prefix": ("STRING", {"default": "lora_pack/lora"}),
            "lora_name": ("STRING", {"default": "trained_lora.safetensors"}),
            "max_dimension": ("INT", {"default": 8192, "min": 256, "max": 16384, "step": 64}),
            "compress": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "pack_it"
    OUTPUT_NODE = True
    CATEGORY = "model/training"

    def pack_it(self, lora, filename_prefix, lora_name, max_dimension, compress):
        import safetensors.torch
        from PIL import Image
        from . import m3ds

        t0 = time.time()
        data = safetensors.torch.save({k: v.detach().contiguous().cpu() for k, v in lora.items()})
        arrays = m3ds.pack(data, lora_name, max_dimension=int(max_dimension), compress=bool(compress))
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = []
        for arr in arrays:
            fname = f"{name}_{counter:05}_.png"
            Image.fromarray(arr, mode="RGB").save(os.path.join(folder, fname), format="PNG", compress_level=1)
            results.append({"filename": fname, "subfolder": subfolder, "type": "output"})
            counter += 1
        print(f"[TrainGlue] packed {lora_name}: {len(data)} bytes -> {len(arrays)} PNG(s), "
              f"first {arrays[0].shape[1]}x{arrays[0].shape[0]}, {time.time() - t0:.1f}s")
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"TrainGlueUNETLoader": TrainGlueUNETLoader, "TrainGluePackLora": TrainGluePackLora}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainGlueUNETLoader": "Load Diffusion Model (no DynamicVRAM, for training)",
    "TrainGluePackLora": "Pack LoRA as PNG (M3DS)",
}
