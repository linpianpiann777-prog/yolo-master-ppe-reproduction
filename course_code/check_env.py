"""Step 0: verify the local environment (CUDA GPU, PyTorch, YOLO-Master code)."""

import platform

import torch

import ultralytics

print(f"Python      : {platform.python_version()}")
print(f"PyTorch     : {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"Ultralytics : {ultralytics.__version__} from {ultralytics.__file__}")
print(f"CUDA usable : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU         : {p.name}, {p.total_memory / 1024**3:.1f} GB")
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    print(f"GPU matmul  : ok, checksum={float((x @ x).abs().mean()):.3f}")
