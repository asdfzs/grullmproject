"""
Leave-one-road-out test: train the GRU on all roads but one, test on the road it never saw. Repeat per road.

    venv\\Scripts\\python.exe gru\\cross_road.py

Data : results\\features\\*.csv (route 165 + normal05)  and  results\\features_newroads\\*.csv (r157, r174, r180)
Label: "long" (stop that will last >= 12 s with no red light) - the GRU's current role.
Out  : gru\\out_cross\\report.txt
Each fold trains a fresh model (a few minutes on the laptop GPU per fold, 4 folds).
"""
import os, sys, glob, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gru_features import featurize, make_model, WIN, DT

_ap = argparse.ArgumentParser()
_ap.add_argument('--win-sec', type=float, default=3.0, help='how many seconds of history the GRU sees (default 3)')
_ap.add_argument('--hz', type=float, default=20.0, help='sampling rate inside the window (20 = every tick)')
A = _ap.parse_args()
STEP = max(1, int(round(20.0 / A.hz)))                  # ticks between samples
NWIN = max(2, int(round(A.win_sec * A.hz)))              # samples per window
TAG = '' if (A.win_sec == 3.0 and A.hz == 20.0) else f'_w{int(A.win_sec)}s_{int(A.hz)}hz'
LONG, STRIDE, THRESH, EPOCHS = 12.0, 5, 0.85, 30
OUT = os.path.join(HERE, 'out_cross'); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def road_of(name):
    if name.startswith('normal05'): return 'normal05'
    if name.startswith('165'): return '165'
    return name.split('_')[0]          # r157 / r174 / r180


def long_labels(stop, red):
    y = np.zeros(len(stop), np.float32); t = 0
    while t < len(stop):
        if stop[t] > 0:
            a = t
            while t < len(stop) and stop[t] > 0: t += 1
            if stop[a:t].max() >= LONG and red[a:t].max() == 0: y[a:t] = stop[a:t] >= 1.0
        else:
            t += 1
    return y


eps = []
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'features', '*.csv')) +
                glob.glob(os.path.join(ROOT, 'results', 'features_newroads', '*.csv'))):
    rows = list(csv.DictReader(open(f, encoding='utf-8')))
    if len(rows) < 20: continue
    name = os.path.basename(f)[:-4]
    stop = np.array([float(r['stop_duration_s']) for r in rows]); red = np.array([int(r['traffic_light_red']) for r in rows])
    eps.append(dict(name=name, road=road_of(name), cls=rows[0]['episode_class'],
                    X=np.array([featurize(r) for r in rows], np.float32), stop=stop, red=red, y=long_labels(stop, red),
                    fstop=np.array([r['front_class'] == 'vehicle_stopped' and float(r['front_dist_m']) < 15 for r in rows])))
roads = sorted(set(e['road'] for e in eps))
print('episodes per road:', {r: sum(e['road'] == r for e in eps) for r in roads}, '| device', dev)


def windows(X, stride):
    """window ending at tick t = samples t-(NWIN-1)*STEP ... t (every STEP ticks), padded with the first row"""
    pad = (NWIN - 1) * STEP
    Xp = np.concatenate([np.repeat(X[:1], pad, 0), X]); idx = range(0, len(X), stride)
    return np.stack([Xp[i:i + pad + 1:STEP] for i in idx]), list(idx)


def events(stop):
    ev, s0 = [], None
    for t, s in enumerate(stop):
        if s > 0 and s0 is None: s0 = t
        if s == 0 and s0 is not None: ev.append((s0, t)); s0 = None
    if s0 is not None: ev.append((s0, len(stop)))
    return [(a, b) for a, b in ev if (b - a) * DT >= 1.0]


def score(test, flags_fn):
    tp = fn = fp = tn = 0; d = []; dP = []
    for e in test:
        fl = flags_fn(e)
        for a, b in events(e['stop']):
            hit = np.nonzero(fl[a:b])[0]; pos = e['y'][a:b].any()
            if pos:
                if len(hit): tp += 1; d.append(hit[0] * DT); (dP.append(hit[0] * DT) if e['cls'] == 'P' else None)
                else: fn += 1
            else:
                fp += len(hit) > 0; tn += len(hit) == 0
    return tp, fn, fp, tn, d, dP


def fmt(name, s):
    tp, fn, fp, tn, d, dP = s
    md = f'{np.mean(d):4.1f}s' if d else '  -  '
    mdp = f'{np.mean(dP):4.1f}s' if dP else '  -  '
    return f'{name:22s} caught {tp}/{tp+fn:<2d} delay {md} (deadlocks only {mdp}) | false alarms {fp}/{fp+tn}'


lines = [f'Leave-one-road-out (label: stop >= 12 s, no red) | GRU history {NWIN} samples x {STEP} ticks = {NWIN*STEP/20:.0f} s', '']
total = {}
for test_road in [r for r in roads if r != 'normal05']:          # normal05 always stays in training
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    tr = [e for e in eps if e['road'] != test_road]; te = [e for e in eps if e['road'] == test_road]
    Xtr = np.concatenate([windows(e['X'], STRIDE)[0] for e in tr])
    ytr = np.concatenate([e['y'][windows(e['X'], STRIDE)[1]] for e in tr])
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    model = make_model(Xtr.shape[-1]).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1., len(ytr) - ytr.sum()) / max(1., ytr.sum()), device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev); yt = torch.tensor(ytr, device=dev)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]; opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
    model.eval()
    for e in te:
        W, _ = windows(e['X'], 1)
        with torch.no_grad():
            e['prob'] = torch.sigmoid(model(torch.tensor((W - mu) / sd, dtype=torch.float32, device=dev))).cpu().numpy()
    lines.append(f'== test road {test_road}  (train: {len(tr)} episodes from {sorted(set(e["road"] for e in tr))}, test: {len(te)})')
    methods = {'rule 1s+stopped car': lambda e: (e['stop'] >= 1) & (e['red'] == 0) & e['fstop'],
               'rule 3s+stopped car': lambda e: (e['stop'] >= 3) & (e['red'] == 0) & e['fstop'],
               'rule 5s': lambda e: (e['stop'] >= 5) & (e['red'] == 0),
               'rule 10s': lambda e: (e['stop'] >= 10) & (e['red'] == 0),
               f'GRU P>={THRESH}': lambda e: e['prob'] >= THRESH}
    for m, fn_ in methods.items():
        s = score(te, fn_); lines.append('  ' + fmt(m, s))
        acc = total.setdefault(m, [0, 0, 0, 0, [], []])
        for k in range(4): acc[k] += s[k]
        acc[4] += s[4]; acc[5] += s[5]
    lines.append('')
    print(f'fold {test_road} done')

lines.append('== ALL FOLDS COMBINED (every road tested only by a model that never saw it)')
for m, s in total.items():
    lines.append('  ' + fmt(m, s))
rep = '\n'.join(lines); print('\n' + rep)
# held-out probabilities for every episode (input of gru/tradeoff.py - threshold / cost analysis)
np.save(os.path.join(OUT, f'probs{TAG}.npy'), np.array([dict(name=e['name'], road=e['road'], cls=e['cls'], prob=e['prob'],
        stop=e['stop'], red=e['red'], y=e['y'], fstop=e['fstop']) for e in eps if 'prob' in e], dtype=object))
print('saved held-out probabilities ->', os.path.join(OUT, f'probs{TAG}.npy'))
open(os.path.join(OUT, f'report{TAG}.txt'), 'w', encoding='utf-8').write(rep + '\n')
