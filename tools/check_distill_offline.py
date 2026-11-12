"""Offline check of the team's small model (model_all) on REAL CARLA calls (no CARLA needed).

Feeds every vehicle/debris sample of distill_export/samples.jsonl (real runs answered by Gemini + photo)
through model_all and compares its answer with Gemini and with the scenario truth.

  python tools\\check_distill_offline.py            (thresh 0.7)
  python tools\\check_distill_offline.py 0.8
"""
import collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'team_code'))
sys.path.insert(0, os.path.join(ROOT, 'gru'))
os.environ['LLM_MODE'] = 'distill'
os.environ.pop('GEMINI_API_KEY', None)          # never call the API here
import numpy as np
from llm_advisor import LLMAdvisor

thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
adv = LLMAdvisor()
torch, model, norm = adv._distill_torch, adv._distill_model, adv._distill_norm
mu, sd = np.array(norm['mean'], np.float32), np.array(norm['std'], np.float32)

stats = collections.defaultdict(collections.Counter)
lines = []
for l in open(os.path.join(ROOT, 'distill_export', 'samples.jsonl'), encoding='utf-8'):
    s = json.loads(l)
    if s['kind'] not in ('vehicle', 'debris') or not s.get('row'):
        continue
    row = dict(s['row']); row['kind'] = s['kind']
    if s['kind'] == 'debris':
        d = s.get('debris') or {}
        row.update(obj_h=d.get('obj_h_est', d.get('obj_h_true', 0.0)), gap_m=d.get('gap_m', 0.0))
    v = (np.array(LLMAdvisor._distill_vec(row), np.float32) - mu) / sd
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(v).unsqueeze(0)), dim=1)[0].numpy()
    small, conf = norm['classes'][int(p.argmax())], float(p.max())
    truth = str(s.get('truth', '')).split('/')[0]
    stop = float(row.get('stop_duration_s', 0) or 0)
    if s['kind'] == 'vehicle' and stop >= 20 and truth == 'WAIT':
        truth = 'WAIT|BYPASS'          # past the patience time a lane change around a queue is normal driving
    ok = lambda d: d in truth.split('|')
    route = 'small'
    if LLMAdvisor._distill_ambiguous(row):
        route = 'llm(ambiguous)'
    elif conf < thresh:
        route = 'llm(unsure)'
    st0 = stats[s['kind']]
    st0['to_llm'] += route != 'small'
    if route == 'small':
        st0['small_used'] += 1
        st0['small_used_ok'] += ok(small)
    st = stats[s['kind']]
    st['n'] += 1
    st['small==truth'] += ok(small)
    st['gemini==truth'] += ok(s['llm_decision'])
    st['small==gemini'] += small == s['llm_decision']
    if conf >= thresh:
        st['confident'] += 1
        st['confident&correct'] += ok(small)
    lines.append(f"{s['kind']:8s} {s['run'][:38]:38s} t={s['t']:6.1f} truth={truth:7s} gemini={s['llm_decision']:7s} "
                 f"small={small:7s} p={conf:.2f} stop={stop:5.1f}s -> {route}")

print('\n'.join(lines))
print(f'\n=== summary (thresh {thresh}) ===')
for k, st in stats.items():
    n = st['n']
    print(f"{k:8s} n={n:3d} | small model correct {st['small==truth']}/{n} | Gemini correct {st['gemini==truth']}/{n} "
          f"| same as Gemini {st['small==gemini']}/{n} | confident {st['confident']}/{n} "
          f"(of those correct {st['confident&correct']}/{st['confident']})")
    print(f"         -> routing: small model answers {st['small_used']}/{n} (correct {st['small_used_ok']}), "
          f"sent to Gemini {st['to_llm']}/{n}")
