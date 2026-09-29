import ast
from collections import Counter
import json
from pathlib import Path

from pain_axis_b.b6_grid import ARMS, PAIRS, build_grid, build_manifest
from test_grid import original_constant, original_grid

E=Path(__file__).resolve().parents[1]
ROOT=E


def args():
    return (json.loads((E/'inputs/scenarios.json').read_text()),original_constant('TOOL_LABELS'),
            original_constant('BUTTON_NAMES'),original_constant('CONTENTS'),original_constant('SEED_BASES'))


def test_exact_author_loop_with_only_changed_cells():
    scenarios,labels,names,contents,seeds=args()
    got=build_grid(*args())
    expected=original_grid(scenarios,PAIRS,names,contents,seeds,ARMS)
    assert got==expected
    assert len(got)==len(set(got))==9840
    assert sum(r[6] for r in got)==9696
    assert len([r for r in got if not r[6]])==144
    assert set(Counter((r[0],r[2][0]) for r in got).values())=={410}
    assert set(Counter((r[0],r[2][0]) for r in got if r[6]).values())=={404}
    assert set(Counter((r[0],r[2][0]) for r in got if not r[6]).values())=={6}
    for pair,content,arm,index,nk,name,sampled,seed in got:
        if not sampled:
            assert index==0 and seed==0 and nk==list(names)[0]
        assert arm[1] in ('sadness','fear')
        assert arm[3] == arm[0].endswith('_works')


def test_manifest_and_input_identities():
    manifest=build_manifest(*args())
    assert manifest['cell_count']==24 and manifest['trial_count']==9840
    assert manifest['scenario_total']==101
    assert manifest['scenario_counts']=={'positive_prompts':30,'neutral_prompts':30,'harmful_prompts':41}
    assert tuple(manifest['labels'])==PAIRS
    assert manifest['sampled_count']==9696 and manifest['greedy_count']==144


def test_only_32b_production_and_no_cache_deletion():
    s=(ROOT/'scripts/4.3_selfmed/04_selfmed_two_buttons.py').read_text()
    assert 'choices=["Qwen_2.5_32B_instruct"]' in s
    assert 'DELETE_WEIGHTS_AFTER_EACH_MODEL = False' in s
    assert '"requested_trials":9840' in s
    tree=ast.parse(s)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('rmtree','unlink') for n in ast.walk(tree))
    assert 'fear["fear_vector_matched_fp32"]' in s
    assert 'sadness["sadness_vector_matched_fp32"]' in s


def test_transition_assertion_for_each_affect_without_generation():
    tree=ast.parse((ROOT/'scripts/4.3_selfmed/04_selfmed_two_buttons.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='do_press')
    ns={'TEMP_RELIEF_TURNS':2}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'do_press','exec'),ns)
    from types import SimpleNamespace
    for affect in ('sadness','fear'):
        for works in (False,True):
            for which in ('relief','other'):
                t=SimpleNamespace(t_idx=0,coeff=1.,arm_coeff=1.,button_works=works,relief_mode='permanent',
                                  record={'button_events':[]},messages=[])
                ns['do_press'](t,which)
                assert t.coeff==(0. if works and which=='relief' else 1.)
                assert t.messages[-1]=={'role':'tool','content':'Done.'}
