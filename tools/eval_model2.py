"""
Check Taeyoung's distilled model (model2) on REAL CARLA trigger moments (no CARLA needed).
    venv\Scripts\python.exe tools\eval_model2.py
For every LLM call of the vehicle scenarios in results\intervention\*\events.jsonl it takes the sensor row
at that moment from results\intervention_features\<run>.csv and compares
  model2 (P(BYPASS) >= 0.5) | "hazard lights -> BYPASS" rule | the answer the car actually got (Gemini/stub)
with what the scenario needs:
  queue / shortqueue / stopgo       -> WAIT is correct at every call (bypassing a queue is the bad mistake)
  broken / stalled                  -> BYPASS is the goal (an early WAIT is acceptable, it is asked again)
Use  --no-model  to only list the trigger moments (no torch needed).
"""
import os, sys, json, glob, csv, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'gru'))
from gru_features import featurize

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', default=r'C:\taeyoungproject\datamodel')
ap.add_argument('--no-model', action='store_true')
args = ap.parse_args()

predict = None
if not args.no_model:
    import torch, torch.nn as nn
    norm = json.load(open(os.path.join(args.model_dir, 'model2_norm.json')))
    net = nn.Sequential(nn.Linear(34, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    sd = torch.load(os.path.join(args.model_dir, 'model2.pt'), map_location='cpu')
    sd = {k.replace('net.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd); net.eval()
    mu, sdv = torch.tensor(norm['mean']), torch.tensor(norm['std']).clamp_min(1e-6)
    def predict(vec):
        with torch.no_grad():
            return float(torch.sigmoid(net(((torch.tensor(vec, dtype=torch.float32) - mu) / sdv)[None]))[0, 0])

def kind_of(run):
    for k in ('shortqueue', 'queue', 'stopgo', 'broken', 'stalled'):
        if f'_{k}_' in run:
            return k
    return None

rows_out, stats = [], {}
for ev in sorted(glob.glob(os.path.join(ROOT, 'results', 'intervention', '*', 'events.jsonl'))):
    run = os.path.basename(os.path.dirname(ev))
    k = kind_of(run)
    if k is None:
        continue
    csvp = os.path.join(ROOT, 'results', 'intervention_features', run + '.csv')
    if not os.path.exists(csvp):
        continue
    feats = list(csv.DictReader(open(csvp, encoding='utf-8')))
    if not feats:
        continue
    evs = [json.loads(l) for l in open(ev, encoding='utf-8')]
    answers = [e for e in evs if e['event'] == 'llm_answer' and e.get('kind') in (None, 'vehicle')]
    trigs = [e for e in evs if e['event'] == 'trigger' and e.get('kind') in (None, 'vehicle')]
    llm = next((e.get('llm') for e in evs if e['event'] == 'start'), '?')
    for i, tr in enumerate(trigs):
        row = min(feats, key=lambda r: abs(float(r['sim_time_s']) - tr['t']))
        vec = featurize(row)
        hz = float(row['front_hazard_on']) > 0
        ans = answers[i]['decision'] if i < len(answers) else '-'
        p = predict(vec) if predict else None
        m2 = ('BYPASS' if p >= 0.5 else 'WAIT') if p is not None else '-'
        rows_out.append((run, k, llm, i + 1, float(row['stop_duration_s']), hz,
                         float(row['front2_dist_m']) < 900, ans, p, m2))

print(f"{'run':44s} {'kind':10s} {'llm':6s} call stop_s haz f2  answer  model2(p)    hazard-rule")
for r in rows_out:
    run, k, llm, c, st, hz, f2, ans, p, m2 = r
    ps = f'{m2:6s}({p:.2f})' if p is not None else '-'
    print(f"{run[:44]:44s} {k:10s} {llm:6s} {c:4d} {st:6.1f} {'Y' if hz else '.':3s} {'Y' if f2 else '.':3s} "
          f"{ans:7s} {ps:12s} {'BYPASS' if hz else 'WAIT'}")

def score(name, pick):
    q = [r for r in rows_out if r[1] in ('queue', 'shortqueue', 'stopgo')]
    bad_q = sum(pick(r) == 'BYPASS' for r in q)
    runs = {}
    for r in rows_out:
        if r[1] in ('broken', 'stalled'):
            runs.setdefault(r[0], []).append(pick(r))
    solved = sum('BYPASS' in v for v in runs.values())
    print(f"  {name:22s} queue-type calls wrongly BYPASS: {bad_q}/{len(q)} | broken/stalled runs reaching BYPASS: {solved}/{len(runs)}")

print('\n== summary (real CARLA trigger moments) ==')
if predict:
    score('model2', lambda r: r[9])
score('hazard rule', lambda r: 'BYPASS' if r[5] else 'WAIT')
score('actual answer (log)', lambda r: r[7])
print('note: broken/stalled runs that ended after the first BYPASS have fewer calls; the log answer came from stub or Gemini (see llm column).')
