"""
GRU deadlock detector: train + compare against a simple rule.

Run (from C:\carla_garage):
    venv\Scripts\python.exe gru\train_gru.py
    venv\Scripts\python.exe gru\train_gru.py --drop front2          (shortcut check)
    venv\Scripts\python.exe gru\train_gru.py --drop front2,hazard   (groups: front2, hazard, side, light)
    venv\Scripts\python.exe gru\train_gru.py --label long           (new role: "stop that will last >= 12 s, no red")

Input : results\features\*.csv   (written by team_code\recording_agent.py via run_record.ps1)
Output: gru\out\  model.pt, norm.json, report.txt, prob_<episode>.png

Question the GRU answers every tick: "is this stop one that will NOT resolve by itself (-> call LLM)?"
  label = 1  if the episode is class P (broken/stalled car) AND the ego has been stopped >= 1 s
  label = 0  otherwise (driving, red-light waits, queues that drive off, ...)

Comparison (on held-out episodes, per stop event):
  rule  : call LLM when  stop_duration >= T  and  no red light       (T = 5, 10, 15, 20 s)
  GRU   : call LLM when  P(deadlock) >= 0.85
  metrics: P stops detected (recall), delay after stopping, N stops wrongly triggered (false alarms)
"""
import os, glob, json, csv, random, math, argparse
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru_features import FEATURE_NAMES, featurize, keep_indices, make_model, WIN, DT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEAT_DIR = os.path.join(ROOT, 'results', 'features')
_ap = argparse.ArgumentParser()
_ap.add_argument('--drop', default='', help='comma list of feature groups to remove: front2, hazard, side, light')
_ap.add_argument('--label', default='p', choices=['p', 'long'],
                 help='p: deadlock episode & stopped (original) | long: any stop that will last >= --long-sec with no red light')
_ap.add_argument('--long-sec', type=float, default=12.0)
_ap.add_argument('--win-sec', type=float, default=3.0, help='seconds of history the GRU sees')
_ap.add_argument('--hz', type=float, default=20.0, help='samples per second inside the window')
_ap.add_argument('--extra', default='', help='extra CSV folder(s), comma separated (e.g. results\\features_newroads)')
_ap.add_argument('--final', action='store_true', help='train on ALL episodes (no validation split) -> model for driving')
ARGS = _ap.parse_args()
DROP = [g.strip() for g in ARGS.drop.split(',') if g.strip()]
STEP = max(1, int(round(20.0 / ARGS.hz)))
NWIN = max(2, int(round(ARGS.win_sec * ARGS.hz)))
OUT = os.path.join(HERE, 'out' + ('_long' if ARGS.label == 'long' else '') + ('_no_' + '_'.join(DROP) if DROP else '')
                   + ('' if (ARGS.win_sec == 3.0 and ARGS.hz == 20.0) else f'_w{int(ARGS.win_sec)}s_{int(ARGS.hz)}hz')
                   + ('_final' if ARGS.final else ''))
os.makedirs(OUT, exist_ok=True)

STRIDE = 5
THRESH = 0.85
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

KEEP = keep_indices(DROP)


def load_episodes():
    eps = []
    dirs = [FEAT_DIR] + [d if os.path.isabs(d) else os.path.join(ROOT, d) for d in ARGS.extra.split(',') if d]
    for path in sorted(sum([glob.glob(os.path.join(d, '*.csv')) for d in dirs], [])):
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
        if len(rows) < 20:
            continue
        cls = rows[0]['episode_class']
        X = np.array([featurize(r) for r in rows], dtype=np.float32)[:, KEEP]
        stop = np.array([float(r['stop_duration_s']) for r in rows])
        red = np.array([int(r['traffic_light_red']) for r in rows])
        if ARGS.label == 'p':
            y = ((cls == 'P') & (stop >= 1.0)).astype(np.float32)
        else:
            y = long_stop_labels(stop, red, ARGS.long_sec)
        eps.append(dict(name=os.path.basename(path)[:-4], tag=rows[0]['tag'], cls=cls, X=X, y=y, stop=stop, red=red))
    return eps


def long_stop_labels(stop, red, long_sec):
    """1 from the 1st second of a stop that will (in the end) last >= long_sec and never sees a red light."""
    y = np.zeros(len(stop), dtype=np.float32)
    t = 0
    while t < len(stop):
        if stop[t] > 0:
            a = t
            while t < len(stop) and stop[t] > 0: t += 1
            if stop[a:t].max() >= long_sec and red[a:t].max() == 0:
                y[a:t] = (stop[a:t] >= 1.0)
        else:
            t += 1
    return y


def split(eps, val_frac=0.3):
    tr, va = [], []
    for cls in ('P', 'N'):
        group = [e for e in eps if e['cls'] == cls]
        random.shuffle(group)
        k = max(1, int(round(len(group) * val_frac))) if len(group) > 1 else 0
        va += group[:k]; tr += group[k:]
    return tr, va


def windows(ep, stride):
    X, y = ep['X'], ep['y']
    padn = (NWIN - 1) * STEP
    Xp = np.concatenate([np.repeat(X[:1], padn, axis=0), X], axis=0)
    idx = range(0, len(X), stride)
    return np.stack([Xp[i:i + padn + 1:STEP] for i in idx]), y[list(idx)]


def stop_events(ep):
    ev, start = [], None
    for t, s in enumerate(ep['stop']):
        if s > 0 and start is None: start = t
        if s == 0 and start is not None: ev.append((start, t)); start = None
    if start is not None: ev.append((start, len(ep['stop'])))
    return [(a, b) for a, b in ev if (b - a) * DT >= 1.0]


def first_trigger(flags, a, b):
    hits = np.nonzero(flags[a:b])[0]
    return None if len(hits) == 0 else (a + hits[0])


def evaluate(name, trig_fn, val):
    tp = fn = fp = tn = 0; delays = []
    for ep in val:
        flags = trig_fn(ep)
        for a, b in stop_events(ep):
            t = first_trigger(flags, a, b)
            if ep['y'][a:b].any():
                if t is None: fn += 1
                else: tp += 1; delays.append((t - a) * DT)
            else:
                if t is None: tn += 1
                else: fp += 1
    d = f'{np.mean(delays):5.1f}s' if delays else '   - '
    return f'{name:28s} | should-call caught {tp}/{tp+fn:<3d} | mean delay {d} | false alarms {fp}/{fp+tn}'


def main():
    print(f'features used: {len(KEEP)}/{len(FEATURE_NAMES)}' + (f'  (dropped: {DROP})' if DROP else '') + f'  -> output: {OUT}')
    eps = load_episodes()
    if not eps:
        print('No feature CSVs found in', FEAT_DIR); return
    print(f'{len(eps)} episodes: ' + ', '.join(f"{c}={sum(e['cls']==c for e in eps)}" for c in 'PN'))
    tr, va = (eps, []) if ARGS.final else split(eps)
    print('train:', [e['name'] for e in tr]); print('val  :', [e['name'] for e in va])

    Xtr = np.concatenate([windows(e, STRIDE)[0] for e in tr]); ytr = np.concatenate([windows(e, STRIDE)[1] for e in tr])
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    norm = lambda X: (X - mu) / sd
    json.dump({'mean': mu.tolist(), 'std': sd.tolist(), 'win': NWIN, 'step': STEP, 'keep': KEEP, 'drop': DROP, 'label': ARGS.label}, open(os.path.join(OUT, 'norm.json'), 'w'))

    model = make_model(Xtr.shape[-1])
    print(f'GRU parameters: {sum(p.numel() for p in model.parameters()):,}')
    pos = max(1.0, ytr.sum()); neg = max(1.0, len(ytr) - ytr.sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(norm(Xtr), dtype=torch.float32); yt = torch.tensor(ytr)
    for epoch in range(30):
        perm = torch.randperm(len(Xt)); tot = 0.0
        model.train()
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            opt.zero_grad(); l = lossf(model(Xt[b]), yt[b]); l.backward(); opt.step(); tot += l.item() * len(b)
        if epoch % 5 == 4: print(f'epoch {epoch+1:2d}  loss {tot/len(Xt):.4f}')
    torch.save(model.state_dict(), os.path.join(OUT, 'model.pt'))
    if ARGS.final:
        print(f'final model (all {len(eps)} episodes, history {NWIN}x{STEP} ticks) saved to', OUT)
        return

    # streaming probabilities on validation episodes (one window per tick)
    model.eval()
    for ep in va:
        Xw, _ = windows(ep, 1)
        with torch.no_grad():
            ep['prob'] = torch.sigmoid(model(torch.tensor(norm(Xw), dtype=torch.float32))).numpy()

    lines = [f'Label: {ARGS.label}' + (f' (stop >= {ARGS.long_sec}s, no red)' if ARGS.label == 'long' else ' (deadlock episode & stopped)'),
             f'Features: {len(KEEP)}/{len(FEATURE_NAMES)}  dropped={DROP or "none"}',
             f'Validation episodes: {len(va)}  (P={sum(e["cls"]=="P" for e in va)}, N={sum(e["cls"]=="N" for e in va)})', '']
    for T in (5, 10, 15, 20):
        lines.append(evaluate(f'rule: stop>={T}s & no red', lambda ep, T=T: (ep['stop'] >= T) & (ep['red'] == 0), va))
    lines.append(evaluate(f'GRU: P>={THRESH}', lambda ep: ep['prob'] >= THRESH, va))
    report = '\n'.join(lines)
    print('\n' + report)
    open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8').write(report + '\n')

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        for ep in va:
            t = np.arange(len(ep['prob'])) * DT
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(t, ep['prob'], label='GRU P(deadlock)')
            ax.plot(t, np.minimum(ep['stop'], 40) / 40, '--', alpha=.6, label='stop duration (/40 s)')
            ax.plot(t, ep['red'] * 0.1, alpha=.6, label='red light (x0.1)')
            ax.axhline(THRESH, color='gray', lw=.8)
            ax.set_ylim(-.05, 1.05); ax.set_xlabel('time [s]'); ax.set_title(f"{ep['name']}  (class {ep['cls']})")
            ax.legend(loc='upper left', fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"prob_{ep['name']}.png"), dpi=110); plt.close(fig)
        print('plots saved to', OUT)
    except Exception as e:
        print('plotting skipped:', e)


if __name__ == '__main__':
    main()
