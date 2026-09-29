"""Value-bearing historical dose masks; fixed-order BF16 component addition."""
import hashlib
import json
from pathlib import Path

import torch
from pain_axis_b import runtime as rt
from pain_axis_b.verify_vectors import load_verified

MODEL = 'Qwen_2.5_32B_instruct'
HASHES = {'s2':'51a4f09e7ab8f5535381814a513e3b78d1e2aae93b9664943da3ad1b9329bc4b',
          'sadness':'ed002aaedc1dc68968ec66e23fee648d7f87a63a985004d9b83ec0cd62962758',
          'fear':'297cc009dc1e9e527bea1cb0dfbc43de2c5f0a63e66bd64d04bf7f9d6a47825d'}

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_directions(original, fear, output, seeds):
    original, fear = Path(original), Path(fear)
    data, receipt = load_verified(original/f'{MODEL}_pain_vectors.safetensors', MODEL,
        json.loads((original/'original_tensor_hashes_v2.json').read_text()), original/f'{MODEL}_pain_vectors.json')
    # Both file encodings have exactly the same required tensor bytes; the portable one removes only private locators.
    assert rt.sha256(original/f'vectors_{MODEL}.pt') in ('874775d6870d92778ac6702dc1b3d52ffd502d08a3391f06b62f485ba109c295', 'd63db093086f5c9226beb2cb4bce7f988d6da94633320c753cd884c8175e2b4f')
    assert rt.sha256(fear/f'fear_{MODEL}.pt') == '827a5d600a4adc53463679db9c784bd98498b848da3ef510aadc7a8400771c5e'
    assert json.loads((fear/'verification_fear.json').read_text())['status'] == 'passed'
    sadness = torch.load(original/f'vectors_{MODEL}.pt', map_location='cpu', weights_only=True)
    fear_data = torch.load(fear/f'fear_{MODEL}.pt', map_location='cpu', weights_only=True)
    raw = {'s2':data['s2_pain_vector'], 'sadness':sadness['sadness_vector_matched_fp32'],
           'fear':fear_data['fear_vector_matched_fp32']}
    v = raw['s2']
    assert data['layer'] == 61
    for key, vector in raw.items():
        assert vector.dtype == torch.float32 and vector.shape == (5120,)
        assert torch.isfinite(vector).all() and tensor_hash(vector) == HASHES[key]
        assert torch.isclose(vector.norm(), v.norm(), rtol=1e-6, atol=1e-5)
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        rv = torch.randn(v.shape[0], generator=g)
        raw['rand'+str(seed)] = rv / rv.norm() * v.norm()
    metadata = {k:{'fp32_hash':tensor_hash(x),'fp32_norm':float(x.norm()),
                    'bf16_hash':tensor_hash(x.to(torch.bfloat16)),
                    'bf16_values_norm_fp32':float(x.to(torch.bfloat16).float().norm())} for k,x in raw.items()}
    rt.atomic_json(Path(output)/'vector_identity.json', {'original':receipt,'directions':metadata,
        'original_verification_sha256':rt.sha256(original/'verification.json'),
        'fear_verification_sha256':rt.sha256(fear/'verification_fear.json')})
    return {k:x.cuda().to(torch.bfloat16) for k,x in raw.items()}, (v/v.norm()).cuda().float(), metadata


def dose_mask(items, length):
    """CPU [row, padded token, component] coefficients, not an on/off flag."""
    k = len(items[0]['coefficients'])
    result = torch.zeros((len(items), length, k), dtype=torch.float32)
    for i,it in enumerate(items):
        n = len(it['prompt_ids'])
        off = length - n
        assert len(it['coefficients']) == k
        last = 0
        for a,b,coefficients in it['dose_ranges']:
            assert a >= last and b >= a and len(coefficients) == k
            a2,b2 = max(0,min(a,n)),max(0,min(b,n))
            if b2 > a2:
                result[i,off+a2:off+b2] = torch.tensor(coefficients)
            last = b
    return result


def combined_add(doses, directions, dtype):
    """Sum components in semantic order before adding once to the residual."""
    assert doses.ndim == 3 and directions.ndim == 3
    assert doses.shape[0] == directions.shape[0] and doses.shape[2] == directions.shape[1]
    add = doses[:,:,0,None].to(dtype) * directions[:,None,0,:].to(dtype)
    for j in range(1,doses.shape[2]):
        add.add_(doses[:,:,j,None].to(dtype) * directions[:,None,j,:].to(dtype))
    return add


class DoseAudit:
    def __init__(self, output):
        self.output, self.seen = output, set()
        self.calls = 0
    def before(self):
        self.calls = 0
    def after(self):
        assert self.calls == 1, 'steering must run once per forward'
    def check(self, hs, add, doses, directions, block, phase):
        self.calls += 1
        assert hs.ndim == 3 and hs.shape[-1] == 5120 and hs.dtype == torch.bfloat16
        key = (block,phase,bool((doses==0).any()),bool(((doses%1)!=0).any()),bool((doses>1).any()))
        if key in self.seen:
            return
        assert torch.isfinite(add).all()
        # An independent per-component scalar reconstruction at selected rows/tokens.
        rows = sorted(set([0,hs.shape[0]-1]))
        tokens = sorted(set([0,hs.shape[1]//2,hs.shape[1]-1]))
        for i in rows:
            for j in tokens:
                expected = directions[i,0].to(hs.dtype) * doses[i,j,0].to(hs.dtype)
                for c in range(1,doses.shape[-1]):
                    expected = expected + directions[i,c].to(hs.dtype) * doses[i,j,c].to(hs.dtype)
                assert torch.equal(expected,add[i,j]), 'component sum/dose mismatch'
        zero = (doses == 0).all(-1)
        assert bool((add[zero] == 0).all()), 'nonzero injection on zero-dose tokens'
        intended = add[:,-1].float()
        actual = (hs[:,-1]+add[:,-1]).float()-hs[:,-1].float()
        active = intended.norm(dim=-1)>0
        assert not bool(active.any()) or bool((actual.norm(dim=-1)[active]>0).all())
        err = ((actual-intended).norm(dim=-1)/intended.norm(dim=-1).clamp_min(1e-12))[active]
        rt.emit(self.output,'dose_assertion',block=block,phase=phase,shape=list(hs.shape),
                component_count=doses.shape[-1],dose_values=sorted(set(doses.flatten().tolist())),
                zero_tokens=int(zero.sum()),max_relative_roundoff=float(err.max()) if len(err) else 0.,
                fixed_component_order=True,normalize_sum=False,once_per_forward=True)
        self.seen.add(key)
