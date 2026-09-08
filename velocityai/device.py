import platform
from _velocityai_c import Device

def is_metal_available() -> bool:
    """Check if Apple Silicon Metal GPU is available."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def is_cuda_available() -> bool:
    """Check if NVIDIA CUDA is available."""
    # Placeholders for CUDA driver check
    return False

def device(dev: str = "cpu") -> Device:
    """Create or resolve a device instance."""
    if isinstance(dev, Device):
        return dev
    return Device.from_string(dev)

__all__ = ["Device", "device", "is_metal_available", "is_cuda_available"]
