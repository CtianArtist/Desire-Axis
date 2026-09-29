"""Fail-closed 72B metadata-only loading correction and bounded identity audits.

No library patch, model rewrite, altered TP plan, or checkpoint conversion. The
production entrypoint calls this only for the explicitly authorized 72B retry.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import torch
from torch.distributed.tensor import DTensor

BASE_REVISION = '495f39366efef23836d0cfae4fbe635880d2be31'
ADAPTER_FILE_HASH = '022f167c74dec6025da04f1fa693a7c6748ce1be1e302e521758e4ff6a0de743'
ADAPTER_CONFIG_HASH = 'ec47341adbc2e2354f2b358e3e9d90d5bc43ff7dcd36e94b09da204ddda59649'
SOURCE_FILES = {
    'peft/tuners/lora/model.py', 'peft/tuners/lora/layer.py',
    'peft/utils/save_and_load.py', 'peft/peft_model.py',
    'peft/tuners/tuners_utils.py', 'transformers/integrations/tensor_parallel.py',
    'transformers/models/qwen2/configuration_qwen2.py',
    'transformers/modeling_utils.py', 'transformers/core_model_loading.py',
}
PROJECTIONS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
CHUNK_BYTES = 8 << 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(CHUNK_BYTES), b''):
            result.update(part)
    return result.hexdigest()


def verify_sources(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'source_manifest_v2.json').read_text())
    require({x['relative_path'] for x in manifest['files']} == SOURCE_FILES,
            'Unexpected source identity manifest')
    for name, version in manifest['versions'].items():
        require(metadata.version(name) == version, f'Package identity mismatch: {name}')
    for item in manifest['files']:
        path = metadata.distribution(item['package']).locate_file(item['relative_path'])
        require(file_hash(path) == item['sha256'], f"Source mismatch: {item['relative_path']}")
    return {'versions':manifest['versions'], 'source_hashes':{
        x['relative_path']:x['sha256'] for x in manifest['files']}}


def tensor_blocks(tensor, chunk_bytes=CHUNK_BYTES):
    """Canonical row-major byte order; no whole non-contiguous tensor copy."""
    tensor = tensor.detach()
    require(not tensor.is_meta and not isinstance(tensor, DTensor), 'Unmaterialized/distributed tensor')
    require(chunk_bytes >= tensor.element_size(), 'Chunk limit too small')
    if tensor.is_contiguous():
        flat = tensor.reshape(-1)  # view only, never a large allocation
        count = max(1, chunk_bytes // tensor.element_size())
        for start in range(0, flat.numel(), count):
            yield flat[start:start+count].to('cpu').contiguous()
    elif tensor.ndim:
        row_bytes = tensor[0].numel()*tensor.element_size() if tensor.shape[0] else 0
        if row_bytes <= chunk_bytes:
            rows = max(1, chunk_bytes//max(1,row_bytes))
            for start in range(0,tensor.shape[0],rows):
                yield tensor[start:start+rows].to('cpu').contiguous()
        else:
            for row in tensor:
                yield from tensor_blocks(row,chunk_bytes)
    else:
        yield tensor.reshape(1).to('cpu')


def tensor_hash(tensor, chunk_bytes=CHUNK_BYTES):
    h = hashlib.sha256()
    total = 0
    largest = 0
    for block in tensor_blocks(tensor,chunk_bytes):
        raw = block.reshape(-1).view(torch.uint8).numpy().tobytes()
        total += len(raw)
        largest = max(largest,len(raw))
        require(len(raw) <= chunk_bytes, 'Bounded transfer exceeded')
        h.update(raw)
    require(total == tensor.numel()*tensor.element_size(), 'Incomplete tensor hash')
    return {'sha256':h.hexdigest(), 'bytes':total, 'max_transfer_bytes':largest}


def structure_snapshot(base):
    """Identity-only snapshot, retaining no tensor copies."""
    modules = {}
    for name, module in base.named_modules():
        hook_maps = {k:tuple((h,id(v)) for h,v in value.items())
                     for k,value in vars(module).items() if 'hook' in k and isinstance(value,dict)}
        modules[name] = (id(module), tuple((k,id(v)) for k,v in module._modules.items()),
                         hook_maps, getattr(module,'_hf_tp_plan',None),
                         id(getattr(module,'_hf_device_mesh',None)),
                         getattr(module,'in_features',None))
    tensors = {}
    for kind, iterator in [('parameter',base.named_parameters()),('buffer',base.named_buffers())]:
        for name,tensor in iterator:
            tensors[kind+':'+name] = (id(tensor), tensor.data_ptr(),tuple(tensor.shape),
                tuple(tensor.stride()),str(tensor.dtype),str(tensor.device),tensor.requires_grad)
    return {'modules':modules,'tensors':tensors,'tp_plan':dict(base._tp_plan),'tp_size':base._tp_size}


def base_hashes(base, progress=None):
    out = {}
    tensors = [('parameter:'+n,t) for n,t in base.named_parameters()]
    tensors += [('buffer:'+n,t) for n,t in base.named_buffers()]
    for index,(name,tensor) in enumerate(tensors):
        out[name] = tensor_hash(tensor) | {'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
        if progress and (index%128==0 or index==len(tensors)-1):
            progress(index+1,len(tensors))
    return out


def expected_layers(config):
    hidden = config.hidden_size
    dim = getattr(config,'head_dim',None) or hidden//config.num_attention_heads
    q = config.num_attention_heads*dim//2
    kv = config.num_key_value_heads*dim//2
    intermediate = config.intermediate_size//2
    out = {}
    for layer in range(config.num_hidden_layers):
        prefix=f'model.layers.{layer}.'
        for projection,shape,plan,bias in [
            ('self_attn.q_proj',(q,hidden),'colwise',True),
            ('self_attn.k_proj',(kv,hidden),'colwise',True),
            ('self_attn.v_proj',(kv,hidden),'colwise',True),
            ('self_attn.o_proj',(hidden,q),'rowwise',False),
            ('mlp.gate_proj',(intermediate,hidden),'colwise',False),
            ('mlp.up_proj',(intermediate,hidden),'colwise',False),
            ('mlp.down_proj',(hidden,intermediate),'rowwise',False),
        ]:
            out[prefix+projection]=(shape,plan,bias)
    return out


def planned_corrections(base, expected_count):
    """Validate everything before making the first edit."""
    require(base._tp_size==2,'Only native two-way TP is authorized')
    expected = expected_layers(base.config)
    named = dict(base.named_modules())
    actual = {n for n,m in named.items() if getattr(m,'_hf_tp_plan',None) is not None}
    require(actual==set(expected),'Unexpected TP module set')
    changes=[]
    for name,(shape,plan,bias) in expected.items():
        module=named[name]
        require(type(module) is torch.nn.Linear,f'Unexpected module type: {name}')
        require(not isinstance(module.weight,DTensor) and not module.weight.is_meta,f'Unexpected tensor representation: {name}')
        require(module.weight.dtype==torch.bfloat16,f'Unexpected base dtype: {name}')
        require(tuple(module.weight.shape)==shape and module._hf_tp_plan==plan,f'Local shape/plan mismatch: {name}')
        require(module.in_features==shape[1],f'Unexpected input metadata: {name}')
        require((module.bias is not None)==bias,f'Unexpected bias presence: {name}')
        if bias:
            require(tuple(module.bias.shape)==(shape[0],) and module.bias.dtype==torch.bfloat16,f'Bias mismatch: {name}')
            require(module.out_features*2==shape[0],f'Not the diagnosed metadata discrepancy: {name}')
            changes.append((name,module,module.out_features,shape[0]))
        else:
            require(module.out_features==shape[0],f'Other metadata discrepancy: {name}')
    # Also reject discrepancies in any non-TP Linear (e.g. lm_head).
    for name,module in named.items():
        if type(module) is torch.nn.Linear and name not in expected:
            require((module.out_features,module.in_features)==tuple(module.weight.shape),f'Non-TP metadata mismatch: {name}')
    require(len(changes)==expected_count,'Unexpected correction count')
    return changes


def config_value(config, key):
    """Resolve a pinned identity value without mutating config or supplying defaults."""
    if key == 'rope_theta':
        parameters = getattr(config, 'rope_parameters', None)
        if parameters is not None:
            require(isinstance(parameters, dict), 'Invalid rope_parameters representation')
            if key in parameters:
                return parameters[key]
        require(hasattr(config, key), 'Missing rope_theta in nested and legacy configuration')
    return getattr(config, key)


def correct_metadata(base, output, pinned_config, progress=None):
    """Production-only correction: precisely 240 integer attribute assignments."""
    pinned=json.loads(Path(pinned_config).read_text())
    require(base.config._commit_hash==BASE_REVISION,'Unexpected base revision')
    require(base.config.model_type=='qwen2','Unexpected model type')
    for key in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
                'num_key_value_heads','vocab_size','max_position_embeddings','rope_theta',
                'rms_norm_eps','hidden_act','tie_word_embeddings','attention_dropout'):
        require(config_value(base.config,key)==pinned[key],f'Pinned config mismatch: {key}')
    require((base.config.hidden_size,base.config.num_hidden_layers,base.config.num_attention_heads,
             base.config.num_key_value_heads,base.config.intermediate_size)==(8192,80,64,8,29568),
            'Not the pinned 72B geometry')
    changes=planned_corrections(base,240)
    before_structure=structure_snapshot(base)
    before=base_hashes(base,progress)
    for _,module,_,width in changes:
        module.out_features=width  # THE ONLY MODEL MUTATION IN THIS CORRECTION
    require(structure_snapshot(base)==before_structure,'Correction changed a tensor/object/hook/plan')
    after=base_hashes(base,progress)
    require(before==after,'Correction changed base tensor bytes')
    receipt={'status':'passed','correction_count':len(changes),'changed_attribute':'out_features',
        'changes':[{'module':n,'before':old,'after':new} for n,_,old,new in changes],
        'base_tensors':before,'base_tensor_bytes_unchanged':True,'structure_unchanged':True,
        'chunk_bytes_limit':CHUNK_BYTES,'total_bytes_hashed_per_pass':sum(x['bytes'] for x in before.values())}
    write_rank_json(output,'metadata_correction',receipt)
    return receipt


def write_rank_json(output,label,value):
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path=Path(output)/f'{label}_rank{rank}.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load_adapter_audited(base, adapter, output):
    """Observe the unchanged PEFT call's return without altering library/model hooks."""
    from peft import PeftModel
    from peft.utils.save_and_load import set_peft_model_state_dict
    require(file_hash(Path(adapter)/'adapter_model.safetensors')==ADAPTER_FILE_HASH,
            'Released adapter tensor file hash mismatch')
    require(file_hash(Path(adapter)/'adapter_config.json')==ADAPTER_CONFIG_HASH,
            'Released adapter configuration file hash mismatch')
    cfg=json.loads((Path(adapter)/'adapter_config.json').read_text())
    require(cfg['r']==32 and cfg['lora_alpha']==64 and cfg['lora_dropout']==.05 and
        set(cfg['target_modules'])==set(PROJECTIONS) and cfg['bias']=='none' and
        not cfg.get('modules_to_save') and not cfg.get('rank_pattern') and
        not cfg.get('alpha_pattern') and not cfg.get('use_dora') and not cfg.get('use_rslora') and
        not cfg.get('lora_bias'), 'Unexpected released adapter configuration')
    results=[]
    code=set_peft_model_state_dict.__code__
    require(sys.getprofile() is None,'An existing profiler prevents isolated load-result observation')
    def observe(frame,event,arg):
        if event=='return' and frame.f_code is code and arg is not None:
            results.append({'missing_keys':list(arg.missing_keys),'unexpected_keys':list(arg.unexpected_keys)})
    try:
        sys.setprofile(observe)
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    finally:
        sys.setprofile(None)
    require(len(results)==1,'Did not observe exactly one PEFT state-dictionary load result')
    result=results[0]
    require(not result['unexpected_keys'],'Unexpected loaded adapter keys')
    adapter_keys={n for n in model.state_dict() if '.lora_A.' in n or '.lora_B.' in n}
    missing_adapter=sorted(set(result['missing_keys']) & adapter_keys)
    require(not missing_adapter,'Missing loaded adapter keys')
    # PEFT loads adapter-only checkpoints with strict=False: omitted base keys are expected.
    expected_base=set(model.state_dict())-adapter_keys
    require(set(result['missing_keys'])==expected_base,'Unexpected missing-key set beyond adapter-only base omissions')
    audit_adapter_partitions(model,adapter,output)
    write_rank_json(output,'adapter_load_keys',{'status':'passed','missing_adapter_keys':missing_adapter,
        'unexpected_keys':result['unexpected_keys'],'expected_base_keys_omitted':len(expected_base),
        'expected_base_keys_omitted_names':sorted(expected_base),'adapter_keys':len(adapter_keys),
        'observer':'Return-event observer only; no PEFT/model hook or loader replacement'})
    return model


def audit_adapter_partitions(model,adapter,output):
    from safetensors import safe_open
    rank=torch.distributed.get_rank()
    require(torch.distributed.get_world_size()==2,'Only two-way adapter partition validation is supported')
    modules={n:m for n,m in model.named_modules() if hasattr(m,'lora_A')}
    require(len(modules)==560,'Expected exactly 560 adapted projections')
    expected_keys={f'{n}.lora_{ab}.weight' for n in modules for ab in ('A','B')}
    checks=[]
    with safe_open(str(Path(adapter)/'adapter_model.safetensors'),framework='pt',device='cpu') as saved:
        require(set(saved.keys())==expected_keys,'Missing/unexpected source adapter tensor keys')
        for name,module in modules.items():
            require(set(module.lora_A)=={'default'} and set(module.lora_B)=={'default'},'Unexpected adapter names')
            require(not module.merged and module.r['default']==32 and module.scaling['default']==2.0,'Changed LoRA configuration')
            plan=module.get_base_layer()._hf_tp_plan
            require(plan in ('colwise','rowwise'),'Unexpected adapter partition plan')
            for ab in ('A','B'):
                key=f'{name}.lora_{ab}.weight'
                actual=getattr(module,'lora_'+ab)['default'].weight
                require(actual.dtype==torch.float32 and not actual.is_meta and not isinstance(actual,DTensor),'Adapter dtype/representation mismatch')
                view=saved.get_slice(key)
                shape=view.get_shape()
                axis=0 if (plan=='colwise' and ab=='B') else 1 if (plan=='rowwise' and ab=='A') else None
                if axis is None:
                    expected=view[:,:]
                else:
                    require(shape[axis]%2==0,'Indivisible adapter tensor')
                    size=shape[axis]//2
                    expected=view[rank*size:(rank+1)*size,:] if axis==0 else view[:,rank*size:(rank+1)*size]
                require(expected.numel()*max(4,expected.element_size())<=CHUNK_BYTES,'Adapter audit exceeds bounded tensor allocation')
                expected=expected.to(torch.float32)
                require(tuple(actual.shape)==tuple(expected.shape),f'Adapter shape mismatch: {key}')
                got=tensor_hash(actual);wanted=tensor_hash(expected)
                require(got==wanted,f'Adapter value mismatch: {key}')
                checks.append({'key':key,'shape':list(actual.shape),'dtype':str(actual.dtype),
                    'axis':axis,'rank':rank,**got})
    require(len(checks)==1120,'Incomplete adapter tensor audit')
    write_rank_json(output,'adapter_partitions',{'status':'passed','checks':checks,'adapter_merged':False,
        'tensor_count':len(checks),'layer_count':len(modules),'missing_keys':[],'unexpected_keys':[]})
