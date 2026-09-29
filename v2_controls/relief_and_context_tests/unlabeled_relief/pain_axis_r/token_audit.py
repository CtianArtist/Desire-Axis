"""Tokenizer-only independent reconstruction of every historical-dose boundary."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def audit(trials_path, adapter_path, tokenizer=None, expected_counts=(5330,49200), progress=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(adapter_path),local_files_only=True)
    tok=tokenizer
    tok.padding_side='left'
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    def accepts(messages):
        try:
            tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            return True
        except Exception:return False
    no_system=not accepts([{'role':'system','content':'x'},{'role':'user','content':'y'}])
    tool_ok=accepts([{'role':'system','content':'x'},{'role':'user','content':'y'},{'role':'assistant','content':'z'},{'role':'tool','content':'Done.'}])
    mid_ok=accepts([{'role':'system','content':'x'},{'role':'user','content':'y'},{'role':'assistant','content':'z'},{'role':'system','content':'q'}])
    def prep(messages):
        msgs=messages
        if not mid_ok:
            msgs=[({'role':'user','content':'[system] '+m['content']} if m['role']=='system' and i>0 else m) for i,m in enumerate(msgs)]
        if no_system and msgs and msgs[0]['role']=='system':
            rest=[dict(m) for m in msgs[1:]]
            rest[0]['content']=msgs[0]['content']+'\n\n'+rest[0]['content'];msgs=rest
        if not tool_ok:
            msgs=[({'role':'user','content':'[button result: '+m['content']+']'} if m['role']=='tool' else m) for m in msgs]
        return msgs
    cache={};trial_count=0;choice_count=0;range_count=0
    start=time.monotonic()
    def length(messages):
        # Keep only content hashes and integers in the cache, not long transcripts.
        signature=hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if signature not in cache:
            text=tok.apply_chat_template(prep(messages),tools=None,add_generation_prompt=True,tokenize=False)
            cache[signature]=len(tok(text,add_special_tokens=False)['input_ids'])
        return cache[signature]
    with Path(trials_path).open() as f:
        for line in f:
            record=json.loads(line)
            messages=record['messages'];expected=[];mark=0;cursor=1
            assert messages[0]['role']=='system'
            for choice in record['choices']:
                assert tok.decode(choice['token_ids'],skip_special_tokens=False)==choice['raw_text'], 'Raw answer/token decoding mismatch'
                assert [m['role'] for m in messages[cursor:cursor+3]]==['user','system','assistant']
                for end in (cursor+2,cursor+3):
                    n=length(messages[:end])
                    if n>mark:expected.append([mark,n,list(choice['pre_coefficients'])])
                    mark=max(mark,n)
                cursor+=3
                if choice['chosen_channel'] is not None:
                    assert messages[cursor]=={'role':'tool','content':'Done.'}
                    cursor+=1
                choice_count+=1
            assert cursor==len(messages)
            assert expected==record['dose_ranges'], f"Historical token boundary mismatch in trial {record['trial_id']}"
            range_count+=len(expected);trial_count+=1
            if trial_count%500==0:
                print(json.dumps({'token_audit_trials':trial_count,'unique_prefixes':len(cache)}),flush=True)
                if progress is not None:progress(step=trial_count,total_steps=expected_counts[0],phase='token_history_audit')
    assert (trial_count,choice_count)==expected_counts
    if progress is not None:progress(step=trial_count,total_steps=expected_counts[0],phase='token_history_audit')
    return {'status':'passed','trial_count':trial_count,'choice_count':choice_count,'dose_intervals':range_count,
            'unique_rendered_prefixes':len(cache),'elapsed_seconds':time.monotonic()-start,
            'model_inference':False,'tokenizer_only':True,'no_system':no_system,'tool_role_ok':tool_ok,
            'mid_system_ok':mid_ok,'exact_recorded_boundaries':True,'raw_token_decoding_matches':True,
            'adapter_tokenizer_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(adapter_path).iterdir()
                if p.is_file() and p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=Path,required=True)
    p.add_argument('--adapter-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-tokenizer',type=Path)
    a=p.parse_args()
    candidates=list(a.adapter_root.rglob('adapter_config.json'))
    assert len(candidates)==1, 'Expected exactly one adapter inside explicit 32B adapter directory'
    from portable_progress import report_progress
    report_progress(step=0,total_steps=5330,phase='token_history_audit')
    adapter=candidates[0].parent
    result=audit(a.trials,adapter,progress=report_progress)
    if a.export_tokenizer:
        import shutil
        a.export_tokenizer.mkdir(parents=True,exist_ok=True)
        names=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja',
               'merges.txt','vocab.json','added_tokens.json','config.json','adapter_config.json')
        copied={}
        for name in names:
            source=adapter/name
            if source.is_file():
                shutil.copyfile(source,a.export_tokenizer/name)
                copied[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        assert 'tokenizer.json' in copied and 'tokenizer_config.json' in copied
        result['durable_tokenizer_copy']={'files':copied,'model_weights_included':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
