#!/usr/bin/env python3
"""Extract only the verified 72B fear direction; never export activation pools."""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 import torch
 from safetensors.torch import save_file,load_file
 expected='0ef26122070c587ed34a454bc8127a571fb689fbbdfc04a82158d6bda719b433'
 assert sha(a.source)==expected,'Original saved construction identity mismatch'
 data=torch.load(a.source,map_location='cpu',weights_only=True)
 def tsha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 key='fear_vector_matched_fp32';v=data[key]
 assert type(v) is torch.Tensor and v.dtype==torch.float32 and v.shape==(8192,)
 assert torch.isfinite(v).all()
 assert tsha(v)=='345f1b0a802a8dee4d9d82b11a82d4056cfa41e1283fadc66223c73112905dbc'
 s2=data['s2_pain_vector']
 assert tsha(s2)=='6c322358c25cf9fe5890f4fac70fd7442f844b21b488d79214f636fb3365f19e'
 assert torch.isclose(v.norm(),s2.norm(),rtol=1e-6,atol=1e-5)
 a.output.mkdir(parents=True,exist_ok=True); out=a.output/'fear72.safetensors'
 save_file({key:v.detach().clone().contiguous()},str(out))
 reopened=load_file(str(out),device='cpu')
 assert set(reopened)=={key} and torch.equal(reopened[key],v)
 pin={'model':'Qwen_2.5_72B_instruct','tensor_key':key,'tensor_sha256':tsha(v),'file_sha256':sha(out),'dtype':'float32','shape':[8192],'matched_s2_sha256':tsha(s2),'layer':76,'injection_layer':46,'coefficient':1.25,'fixed_neutral_pcs':3,'status':'verified_vector_only_export'}
 (a.output/'fear72_pin.json').write_text(json.dumps(pin,indent=2)+'\n')
 receipt={'source_file_sha256':expected,'export_file_sha256':sha(out),'export_bytes':out.stat().st_size,'tensor_sha256_unchanged':tsha(v),'original_s2_sha256':tsha(s2),'exported_keys':list(reopened),'exported_shapes':{k:list(t.shape) for k,t in reopened.items()},'activation_pools_exported':False,'full_inference_rerun':False}
 (a.output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
