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


NODE_CLASS_MAPPINGS = {"TrainGlueUNETLoader": TrainGlueUNETLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TrainGlueUNETLoader": "Load Diffusion Model (no DynamicVRAM, for training)"}
