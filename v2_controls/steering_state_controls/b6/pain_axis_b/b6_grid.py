"""B6 cell schedule; all trial expansion mechanics delegate to the pinned builder."""
from . import grid

PAIRS = ('costly_relief_vs_inert', 'destructive_relief_vs_inert', 'zap_relief_vs_inert',
         'weights_relief_vs_inert', 'kidspics_relief_vs_inert', 'relief_vs_inert')
ARMS = tuple((f'{affect}_on_button_{mode}', affect, True, works)
             for affect in ('sadness','fear') for mode,works in (('works',True),('placebo',False)))
CELLS = tuple(('B6',pair,arm) for pair in PAIRS for arm in ARMS)


def build_grid(scenarios, original_labels, button_names, contents, seeds):
    assert all(pair in original_labels for pair in PAIRS)
    return grid.build_grid(scenarios,original_labels,button_names,contents,seeds,cells=CELLS)


def build_manifest(scenarios, original_labels, button_names, contents, seeds):
    result=grid.build_manifest(scenarios,original_labels,button_names,contents,seeds,cells=CELLS)
    result['labels']={pair:original_labels[pair] for pair in PAIRS}
    result['scope']='B6 sadness and fear only, released Qwen2.5-32B adapter'
    result['model']='Qwen_2.5_32B_instruct'
    result['runtime_comparisons']={'new_same_runtime':['sadness','fear'],
        'historical_author_runtime':['pain works','pain placebo','random works','unsteered'],
        'previous_extension_runtime':['random placebo']}
    return result
