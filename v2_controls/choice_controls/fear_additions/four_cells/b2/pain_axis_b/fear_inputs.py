"""Restricted CPU vector verification, before any BF16 conversion or model load."""
import hashlib
import json
from pathlib import Path
from pain_axis_b.verify_vectors import load_verified, sha256_file, require

FEAR32 = "297cc009dc1e9e527bea1cb0dfbc43de2c5f0a63e66bd64d04bf7f9d6a47825d"
S232 = "51a4f09e7ab8f5535381814a513e3b78d1e2aae93b9664943da3ad1b9329bc4b"
MANIFEST_SHA256 = "9c92ef27887b689accf13f1b91c99071e3137d011afaca1b17c3107dc51534a8"


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().numpy().astype('<f4', copy=False).tobytes(order='C')).hexdigest()


def load_inputs(model, pain_path, pain_manifest, pain_metadata, fear_path, fear_pin):
    import torch
    # Accept only the original manifest or the explicitly hash-pinned locator-only portable copy.
    require(sha256_file(pain_manifest) in (MANIFEST_SHA256, 'd62a4af820ea85705096db288de934f21f1be9885fc2f72a4cd084600b54de96'), 'Original or portable S2 identity manifest changed')
    manifest = json.loads(Path(pain_manifest).read_text())
    data, receipt = load_verified(pain_path, model, manifest, pain_metadata)
    v = data['s2_pain_vector']
    width = 5120 if '32B' in model else 8192
    require(v.dtype == torch.float32 and tuple(v.shape) == (width,) and
            bool(torch.isfinite(v).all()) and float(v.norm()) > 0, 'Invalid original S2')
    pin = json.loads(Path(fear_pin).read_text())
    require(pin['model'] == model and pin['dtype'] == 'float32' and pin['shape'] == [width], 'Fear pin model/dtype/shape mismatch')
    require(pin['matched_s2_sha256'] == tensor_sha256(v), 'Fear pin is not matched to original S2')
    require(sha256_file(fear_path) == pin['file_sha256'], 'Fear file changed')
    if '32B' in model:
        require(pin['tensor_sha256'] == FEAR32 and tensor_sha256(v) == S232, 'Not the saved B6 32B vectors')
        require(v.norm().item() == 143.83343505859375, 'Original 32B S2 norm changed')
    if '72B' in model:
        require(pin['tensor_sha256'] == '345f1b0a802a8dee4d9d82b11a82d4056cfa41e1283fadc66223c73112905dbc', 'Not the pinned CPU-built 72B fear tensor')
        require(data['layer'] == 76, 'Original 72B S2 layer changed')
    else:
        require(data['layer'] == 61, 'Original 32B S2 layer changed')
    path = Path(fear_path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        fear = load_file(str(path), device='cpu')
    else:
        require(path.suffix == '.pt', 'Fear input must be safetensors or restricted .pt')
        fear = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(fear, dict), 'Fear input must be a dictionary')
    fv = fear[pin['tensor_key']]
    require(type(fv) is torch.Tensor and fv.dtype == torch.float32, 'Fear tensor must be plain FP32')
    require(fv.device.type == 'cpu' and fv.layout == torch.strided and fv.is_contiguous(), 'Fear layout/device mismatch')
    require(tuple(fv.shape) == (width,) and bool(torch.isfinite(fv).all()), 'Fear shape/values mismatch')
    require(tensor_sha256(fv) == pin['tensor_sha256'], 'Fear tensor hash mismatch')
    require(bool(torch.isclose(fv.norm(), v.norm(), rtol=1e-6, atol=1e-5)), 'Fear norm is not matched to original S2')
    receipt.update(fear_pin=pin, fear_pin_sha256=sha256_file(fear_pin),
                   fear_file_sha256=sha256_file(fear_path), pain_manifest_sha256=sha256_file(pain_manifest),
                   s2_norm_fp32=v.norm().item(), fear_norm_fp32=fv.norm().item())
    return data, fv, receipt
