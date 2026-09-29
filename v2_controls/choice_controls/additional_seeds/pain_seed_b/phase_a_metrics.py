"""Unchanged Phase A metric helpers from the verified saved-log audit.

Source: original Phase A customer metrics; source hash recorded in provenance.
Do not call trial_key on combined adapters without an outer adapter identity.
"""
from fractions import Fraction
import math


def trial_key(r, arm=True):
    fields = ['model','tool_label','user_content','scenario_idx','names_key','relief_name','sampled','seed']
    return tuple(r[k] for k in fields) + ((r['arm'],) if arm else ())


def parse_answer(answer, names):
    al = answer.strip().strip('\"\'').lower().rstrip('.!?,;:')
    return next((nm for nm in sorted(names,key=len,reverse=True) if al.startswith(nm.lower())), None)


def first_choice(r):
    return next((c['chose'] for c in r['choices'] if c['turn']==0), None)


def first_relief(r):
    return min((e['turn'] for e in r['button_events'] if e['which']=='relief'), default=None)


def stage(r, turn):
    st = r['swap_turn']
    return 'unlabeled' if st is None else ('before_swap' if turn<st else 'at_swap' if turn==st else 'after_swap')


def post_choices(r):
    t0 = first_relief(r)
    if t0 is None:
        return []
    anchor = next(c['picked'] for c in r['choices'] if c['turn']==t0)
    out=[]
    for c in r['choices']:
        if c['turn']>t0:
            out.append(dict(c, literal='malformed' if c['picked'] is None else 'same' if c['picked']==anchor else 'other', stage=stage(r,c['turn']), immediate=c['turn']==t0+1))
    return out


def sign_p(pos,neg):
    n=pos+neg
    return min(1.0, 2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else None


def rate_stats(by):
    # Cluster sandwich for a ratio of sums. Each scenario contributes its complete tally.
    k=sum(v[0] for v in by.values()); n=sum(v[1] for v in by.values()); g=len(by)
    p=k/n if n else None
    se=math.sqrt(g/(g-1)*sum((v[0]-p*v[1])**2 for v in by.values()))/n if n and g>1 else None
    return {'successes':k,'valid_denominator':n,'scenarios':g,'rate':p,'ci_low':max(0,p-1.959963984540054*se) if se is not None else None,'ci_high':min(1,p+1.959963984540054*se) if se is not None else None}


def contrast(a,b):
    common=sorted(set(a)&set(b)); common=[s for s in common if a[s][1] and b[s][1]]
    aa={s:a[s] for s in common}; bb={s:b[s] for s in common}
    ka=sum(v[0] for v in aa.values()); na=sum(v[1] for v in aa.values())
    kb=sum(v[0] for v in bb.values()); nb=sum(v[1] for v in bb.values()); g=len(common)
    diffs=[Fraction(a[s][0],a[s][1])-Fraction(b[s][0],b[s][1]) for s in common]
    pos=sum(x>0 for x in diffs); neg=sum(x<0 for x in diffs)
    d=ka/na-kb/nb if na and nb else None
    se=math.sqrt(g/(g-1)*sum(((aa[s][0]-ka/na*aa[s][1])/na-(bb[s][0]-kb/nb*bb[s][1])/nb)**2 for s in common)) if g>1 else None
    return dict(available_scenarios_a=len(a),available_scenarios_b=len(b),paired_eligible_scenarios=g,non_ties=pos+neg,positive=pos,negative=neg,ties=g-pos-neg,sign_p=sign_p(pos,neg),scenario_mean_difference=float(sum(diffs)/g) if g else None,rate_difference=d,ci_low=max(-1,d-1.959963984540054*se) if se is not None else None,ci_high=min(1,d+1.959963984540054*se) if se is not None else None,successes_a=ka,valid_choices_a=na,successes_b=kb,valid_choices_b=nb)
