"""
Test an already-trained GRU on NEW data it never saw (e.g. other roads) and compare with the rule.

    venv\\Scripts\\python.exe gru\\eval_gru.py --model gru\\out_long --data results\\features_newroads

--model : folder with model.pt + norm.json (written by train_gru.py; label type is read from norm.json)
--data  : folder with feature CSVs (run_record.ps1 ... -OutDir features_newroads)
Output  : <model>\\eval_<data folder name>\\report.txt and prob_<episode>.png
"""
import os, sys, glob, csv, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru_features import featurize, make_model, DT

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--thresh', type=float, default=0.85)
ap.add_argument('--long-sec', type=float, default=12.0)
a = ap.parse_args()

meta = json.load(open(os.path.join(a.model, 'norm.json')))
keep, win, label = meta.get('keep'), meta.get('win', 60), meta.get('label', 'p')
step = meta.get('step', 1)
mu, sd = np.array(meta['mean'], np.float32), np.array(meta['std'], np.float32)
model = make_model(len(keep))
model.load_state_dict(torch.load(os.path.join(a.model, 'model.pt'), map_location='cpu', weights_only=True))
model.eval()
out = os.path.join(a.model, 'eval_' + os.path.basename(os.path.normpath(a.data)))
os.makedirs(out, exist_ok=True)


def stop_events(stop):
    ev, s0 = [], None
    for t, s in enumerate(stop):
        if s > 0 and s0 is None: s0 = t
        if s == 0 and s0 is not None: ev.append((s0, t)); s0 = None
    if s0 is not None: ev.append((s0, len(stop)))
    return [(x, y) for x, y in ev if (y - x) * DT >= 1.0]


eps = []
for f in sorted(glob.glob(os.path.join(a.data, '*.csv'))):
    rows = list(csv.DictReader(open(f, encoding='utf-8')))
    if len(rows) < 20:
        continue
    cls = rows[0]['episode_class']
    X = (np.array([featurize(r) for r in rows], np.float32)[:, keep] - mu) / sd
    stop = np.array([float(r['stop_duration_s']) for r in rows])
    red = np.array([int(r['traffic_light_red']) for r in rows])
    padn = (win - 1) * step
    Xp = np.concatenate([np.repeat(X[:1], padn, 0), X])
    W = np.stack([Xp[i:i + padn + 1:step] for i in range(len(X))])
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(W))).numpy()
    evs = stop_events(stop)
    pos = []
    for x, y in evs:
        if label == 'p':
            pos.append(cls == 'P')
        else:
            pos.append(stop[x:y].max() >= a.long_sec and red[x:y].max() == 0)
    first = next((r for r in rows if float(r['stop_duration_s']) >= 1.0), None)
    eps.append(dict(name=os.path.basename(f)[:-4], cls=cls, prob=prob, stop=stop, red=red, evs=evs, pos=pos,
                    info=(first['front_width_m'], first['front_hazard_on'], first['front2_dist_m']) if first else None,
                    haz=max(int(r['front_hazard_on']) for r in rows)))


def score(name, flags_fn):
    tp = fn = fp = tn = 0; delays = []
    for e in eps:
        fl = flags_fn(e)
        for (x, y), p in zip(e['evs'], e['pos']):
            hits = np.nonzero(fl[x:y])[0]
            if p:
                if len(hits): tp += 1; delays.append(hits[0] * DT)
                else: fn += 1
            else:
                if len(hits): fp += 1
                else: tn += 1
    d = f'{np.mean(delays):5.1f}s' if delays else '   - '
    return f'{name:26s} | should-call caught {tp}/{tp+fn:<3d} | mean delay {d} | false alarms {fp}/{fp+tn}'


lines = [f'Model: {a.model} (label={label}, features={len(keep)})', f'Test data: {a.data}  ({len(eps)} episodes, never used in training)', '']
lines.append('Per episode: stop events (s) -> should call?  | first-stop front width/hazard/2nd car | GRU max prob')
for e in eps:
    evs = ', '.join(f"{(y-x)*DT:.1f}{'*' if p else ''}" for (x, y), p in zip(e['evs'], e['pos']))
    lines.append(f"  {e['name'][:32]:32s} {e['cls']}  stops[{evs}]  front={e['info']} haz_any={e['haz']}  maxP={e['prob'].max():.2f}")
lines.append('  (* = should call LLM)')
lines.append('')
for T in (5, 10, 15, 20):
    lines.append(score(f'rule: stop>={T}s & no red', lambda e, T=T: (e['stop'] >= T) & (e['red'] == 0)))
lines.append(score(f'GRU: P>={a.thresh}', lambda e: e['prob'] >= a.thresh))
rep = '\n'.join(lines)
print(rep)
open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(rep + '\n')
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for e in eps:
        t = np.arange(len(e['prob'])) * DT
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, e['prob'], label='GRU P'); ax.plot(t, np.minimum(e['stop'], 40) / 40, '--', alpha=.6, label='stop (/40s)')
        ax.plot(t, e['red'] * .1, alpha=.6, label='red (x0.1)'); ax.axhline(a.thresh, color='gray', lw=.8)
        ax.set_ylim(-.05, 1.05); ax.set_title(f"{e['name']} (class {e['cls']}) - NEW ROAD"); ax.legend(fontsize=8, loc='upper left')
        fig.tight_layout(); fig.savefig(os.path.join(out, f"prob_{e['name']}.png"), dpi=110); plt.close(fig)
    print('plots ->', out)
except Exception as ex:
    print('plotting skipped:', ex)
