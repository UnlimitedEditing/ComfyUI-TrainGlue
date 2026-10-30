# ComfyUI-TrainGlue

Glue for training LoRAs inside Graydient jobs with ComfyUI core's `TrainLoraNode`.

`TrainGlueUNETLoader` is `UNETLoader` with `disable_dynamic=True`, so the model is a plain `ModelPatcher`
instead of a comfy-aimdo DynamicVRAM one (which crashes during backward on Graydient runners).

`TrainGluePackLora` packs a LORA_MODEL into lossless M3DS PNG(s) (same container as Meshsmuggler); decode with `python unpack_lora_png.py out_dir chunk.png ...`.
