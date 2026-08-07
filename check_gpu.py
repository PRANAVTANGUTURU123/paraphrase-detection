"""CUDA sanity check — run before starting real training on the GPU box."""

import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA NOT available. On Kaggle: Settings -> Accelerator -> GPU, "
        "then restart the session."
    )
print("device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
x = torch.rand(1000, 1000, device="cuda") @ torch.rand(1000, 1000, device="cuda")
print("test matmul on GPU: OK", x.shape)
