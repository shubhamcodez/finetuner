"""Model compression and deployment-target planning."""

from finetuner.quantization.specs import (
    DeviceTarget,
    QuantizationBackend,
    QuantizationConfig,
    backend_specs,
)

__all__ = ["DeviceTarget", "QuantizationBackend", "QuantizationConfig", "backend_specs"]
