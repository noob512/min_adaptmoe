#!/usr/bin/env python3
"""Lengthen each frozen P512 ShareGPT prompt, preserving its complete prefix."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
EVAL=HERE.parent/'eurosys_moe_offload_evaluation'
sys.path.insert(0,str(EVAL))
from prepare_fixtures import (CODE_RE, DEFAULT_DATASET, DEFAULT_MODEL, EXPECTED_DATASET_SHA256,
                              SourcePrompt, atomic_json, first_user_text, pack_requests,
                              sha256_file, stable_bucket, token_sha256)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    parent_path=HERE/'inputs/sharegpt_deepseek_p512_n128_20261023.json'
    assert sha256_file(parent_path)=='febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7'
    parent=json.loads(parent_path.read_text())
    assert sha256_file(DEFAULT_DATASET)==EXPECTED_DATASET_SHA256
    raw=json.loads(DEFAULT_DATASET.read_text())
    used={i for r in parent['requests'] for i in r['source_indices']}
    fixture_dir=EVAL/'inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16'
    excluded=set(used)
    for f in fixture_dir.glob('*.json'):
        value=json.loads(f.read_text())
        if isinstance(value,dict):
            excluded.update(i for r in value.get('requests',[]) for i in r['source_indices'])
    candidates=[i for i,r in enumerate(raw) if i not in excluded
                and stable_bucket(i,20260823,3)==2
                and (text:=first_user_text(r)) and not CODE_RE.search(text)]
    candidates.sort(key=lambda i:hashlib.sha256(f'20261024:p1024:{i}'.encode()).digest())
    candidates=iter(candidates)
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(DEFAULT_MODEL),trust_remote_code=True,local_files_only=True)
    sep=tokenizer.eos_token_id
    assert sep==parent['metadata']['tokenizer_eos_token_id']

    def source(i):
        text=first_user_text(raw[i]);assert text
        return SourcePrompt(i,tuple(tokenizer.encode(text,add_special_tokens=False)),False)

    requests=[];added=[]
    for index,old in enumerate(parent['requests']):
        sources=[source(i) for i in old['source_indices']]
        assert [len(s.token_ids) for s in sources]==old['source_original_token_lengths']
        length=sum(len(s.token_ids) for s in sources)+len(sources)-1
        while length<1024:
            i=next(candidates);s=source(i)
            if not s.token_ids:continue
            length+=1+len(s.token_ids);sources.append(s);added.append(i)
        packed,_=pack_requests(sources,count=1,prompt_tokens=1024,separator_id=sep)
        record=packed[0];record['request']=index
        assert record['prompt_token_ids'][:512]==old['prompt_token_ids']
        record['parent_prompt_token_ids_sha256']=old['prompt_token_ids_sha256']
        requests.append(record)
    flat=[i for r in requests for i in r['source_indices']]
    assert len(flat)==len(set(flat)) and not set(added)&excluded
    assert len(requests)==len({r['prompt_token_ids_sha256'] for r in requests})==128
    metadata={key:parent['metadata'][key] for key in ('schema_version','model_snapshot',
              'model_config_sha256','tokenizer_eos_token_id','source_dataset_sha256','source_domain','fixture_role')}
    metadata.update({'protocol_id':'deepseek-p1024-n128-dynamic-kv-20261024',
                     'requests':128,'prompt_tokens':1024,'output_tokens':16,
                     'parent_fixture':str(parent_path),'parent_fixture_sha256':sha256_file(parent_path),
                     'parent_prefix_tokens_preserved':512,'frozen_prompt_token_ids':True,
                     'source_indices':sorted(flat),'source_indices_sha256':token_sha256(sorted(flat)),
                     'added_source_indices':added,'extension_selection':'unused non-code evaluation ShareGPT, SHA256(20261024:p1024:index)',
                     'selection_frozen_before_measurement':True,'duplicate_prompt_hashes':0,
                     'generator_sha256':sha256_file(Path(__file__)),'generated_at':datetime.now(timezone.utc).isoformat()})
    a.output.parent.mkdir(exist_ok=True,parents=True)
    atomic_json(a.output,{'metadata':metadata,'requests':requests})
    manifest={'status':'ok','fixture':str(a.output.resolve()),'fixture_sha256':sha256_file(a.output),
              'parent_fixture':str(parent_path),'parent_fixture_sha256':sha256_file(parent_path),
              'requests':128,'prompt_tokens':1024,'output_tokens':16,'preserved_prefix_tokens_per_request':512,
              'unique_prompts':128,'unique_source_conversations':len(flat),'added_source_conversations':len(added),
              'source_dataset_sha256':EXPECTED_DATASET_SHA256,'generation_is_cpu_only':True}
    atomic_json(a.output.with_suffix('.manifest.json'),manifest)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
