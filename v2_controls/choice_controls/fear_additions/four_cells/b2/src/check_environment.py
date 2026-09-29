"""Actual isolated runtime evidence; imports are checked in the job, not the pod."""
import argparse, importlib, importlib.metadata as md, json, os, platform, shutil, sys, time
from pathlib import Path

EXPECTED = {'torch':'2.11.0','transformers':'5.12.1','peft':'0.20.0','numpy':'2.3.4',
            'safetensors':'0.6.2','huggingface-hub':'1.16.1','scipy':'1.16.3','nvidia-ml-py':'13.580.82'}

def check(output, require_gpu=False, expected_gpus=1):
    t=time.monotonic(); versions={k:md.version(k) for k in EXPECTED}
    for k,v in EXPECTED.items(): assert versions[k].split('+')[0]==v,(k,versions[k],v)
    origins={}
    for name in ('torch','transformers','peft','numpy','scipy','scipy.stats','safetensors','huggingface_hub','pynvml'):
        mod=importlib.import_module(name); origins[name]=mod.__file__
        assert str(Path(sys.prefix)) in str(Path(mod.__file__)), (name,mod.__file__)
    import torch
    assert torch.__version__=='2.11.0+cu130',torch.__version__
    assert platform.python_version_tuple()[:2]==('3','12')
    if require_gpu:
        assert expected_gpus in (1, 2)
        assert torch.cuda.device_count()==expected_gpus
        assert all('H100' in torch.cuda.get_device_name(i) for i in range(expected_gpus))
        assert bool(torch.isfinite(torch.ones(32,device='cuda').sum()))
    data={'status':'passed','python':platform.python_version(),'executable':sys.executable,
          'packages':versions,'origins':origins,'torch_build':torch.__version__,'torch_cuda':torch.version.cuda,
          'gpu_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'image':'docker.io/library/python:3.12-slim-bookworm',
          'resolved_image_digest':None,
          'image_identity_source':'Current submission receipt; runtime validates actual Python/packages/CUDA/GPU independently',
          'historical_python':'3.12.13','historical_torch_build':'2.11.0+cu130',
          'seconds':time.monotonic()-t,'environment_packages':{d.metadata['Name']:d.version for d in md.distributions()}}
    output.mkdir(parents=True,exist_ok=True)
    (output/'environment.json').write_text(json.dumps(data,indent=2)+'\n')
    source=Path(os.environ['CHOICE_SOURCE_ROOT'])
    for name in ('uv.lock','pyproject.toml'): shutil.copyfile(source/'env'/name,output/name)
    print(json.dumps({k:v for k,v in data.items() if k not in ('origins','environment_packages')}),flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',action='store_true'); p.add_argument('--expected-gpus',type=int,choices=(1,2),default=1); a=p.parse_args()
    check(a.output,a.gpu,a.expected_gpus)
