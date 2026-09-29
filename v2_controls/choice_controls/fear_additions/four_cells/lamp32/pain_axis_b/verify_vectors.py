"""Verify compatible AUTHOR-SUPPLIED originals; never repair or reconstruct them.

Only torch.load(weights_only=True) with default trusted types, or safetensors,
is permitted here. The legacy original .pt files are never loaded by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_verified(candidate, model, manifest, metadata=None):
    """Return unchanged tensors and a receipt; errors propagate, no fallback loader."""
    import torch
    candidate = Path(candidate)
    expected = manifest['models'][model]
    if candidate.suffix == '.safetensors':
        from safetensors.torch import load_file
        require(metadata is not None, 'safetensors requires an author-supplied JSON metadata file')
        data = dict(load_file(str(candidate), device='cpu'))
        header = json.loads(Path(metadata).read_text())
        require(isinstance(header, dict), 'metadata must be a dictionary')
        require(not (set(header) & set(data)), 'metadata cannot replace a tensor entry')
        data.update(header)
    else:
        require(candidate.suffix == '.pt', 'Expected restricted-loader-compatible .pt or safetensors')
        require(metadata is None, '.pt must contain its own plain metadata')
        data = torch.load(candidate, weights_only=True, map_location='cpu')
    require(isinstance(data, dict), 'Expected dictionary')
    require(type(data.get('layer')) is int, 'layer must be a plain Python int, not a scalar object or bool')
    require(data['layer'] == expected['metadata']['layer'], 'layer mismatch')
    for key in ('model','model_name'):
        if key in data:
            require(data[key] == model, f'{key} mismatch')
    if 'extraction' in data:
        require(data['extraction'] == 'final_token', 'extraction mismatch')
    observed = {}
    for key,spec in expected['tensors'].items():
        tensor = data.get(key)
        require(type(tensor) is torch.Tensor, f'{key}: expected plain tensor')
        require(tensor.dtype == torch.float32, f'{key}: dtype changed')
        require(list(tensor.shape) == spec['shape'], f'{key}: shape changed')
        require(tensor.device.type == 'cpu' and tensor.layout == torch.strided, f'{key}: layout/device mismatch')
        require(tensor.is_contiguous(), f'{key}: contiguous layout required')
        # No numerical normalization, cast, or re-creation of either source vector.
        raw = tensor.detach().numpy().astype('<f4', copy=False).tobytes(order='C')
        digest = hashlib.sha256(raw).hexdigest()
        require(len(raw) == spec['bytes'], f'{key}: byte count mismatch')
        require(digest == spec['sha256'], f'{key}: raw tensor bytes changed')
        observed[key] = {'dtype':'float32','shape':list(tensor.shape),'bytes':len(raw),
                         'sha256':digest,'matches_original_bytes':True}
    receipt = {'version':1,'status':'passed','model':model,
        'candidate_path':str(candidate.resolve()),'candidate_file_sha256':sha256_file(candidate),
        'candidate_file_bytes':candidate.stat().st_size,
        'original_source_ref':expected['source_ref'],'original_file_sha256':expected['source_file_sha256'],
        'layer':data['layer'],'layer_type':'plain Python int','tensors':observed,
        'loader':'safetensors' if candidate.suffix == '.safetensors' else 'torch.load(weights_only=True), no added trusted globals',
        'conversion_performed':False,'vector_values_modified':False}
    if metadata is not None:
        receipt['metadata_file_sha256'] = sha256_file(metadata)
    return data, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    _, receipt = load_verified(args.candidate,args.model,manifest,args.metadata)
    receipt['manifest_sha256'] = sha256_file(args.manifest)
    require(not args.receipt.exists(), 'Refusing to overwrite existing verification receipt')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'passed','model':args.model,'receipt':str(args.receipt)}))


if __name__ == '__main__':
    main()
