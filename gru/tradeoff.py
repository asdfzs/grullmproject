"""
"Call early" vs "don't call uselessly": pick the LLM-call policy by COST instead of by feeling.

    venv\\Scripts\\python.exe gru\\tradeoff.py          (after gru\\cross_road.py; no GPU needed)

Input : gru\\out_cross\\probs.npy  (GRU probabilities on roads the model never saw)
Cost  : total = (seconds stuck before the LLM is called, summed over stops that SHOULD call; a missed one costs
        its whole duration) + C x (number of useless calls).  C = "one LLM call is as bad as C seconds of waiting".
Policies compared:
  rule  T s                : stop >= T s and no red light
  rule  T s + stopped car  : same, and a stopped vehicle within 15 m ahead
  GRU   P >= th            : fixed threshold
  GRU   schedule hi->lo    : strict right after stopping (P >= hi), threshold drops linearly to lo at t2 seconds
Output: gru\\out_cross\\tradeoff.txt, tradeoff.png
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out_cross')
DT = 0.05
TAG = sys.argv[1] if len(sys.argv) > 1 else ''      # e.g. _w20s_4hz  (matches cross_road.py output)
eps = list(np.load(os.path.join(OUT, f'probs{TAG}.npy'), allow_pickle=True))


def events(stop):
    ev, s0 = [], None
    for t, s in enumerate(stop):
        if s > 0 and s0 is None: s0 = t
        if s == 0 and s0 is not None: ev.append((s0, t)); s0 = None
    if s0 is not None: ev.append((s0, len(stop)))
    return [(a, b) for a, b in ev if (b - a) * DT >= 1.0]


EV = [(e, a, b, bool(e['y'][a:b].any())) for e in eps for a, b in events(e['stop'])]
n_pos = sum(p for *_, p in EV); n_neg = len(EV) - n_pos
n_min = sum(len(e['stop']) * DT for e in eps if not e['y'].any()) / 60.0   # minutes of episodes that never need a call


def evaluate(flag_fn):
    delays, missed, false = [], 0, 0
    for e, a, b, pos in EV:
        hit = np.nonzero(flag_fn(e, a, b))[0]
        if pos:
            if len(hit): delays.append(hit[0] * DT)
            else: missed += 1; delays.append((b - a) * DT)
        elif len(hit):
            false += 1
    return dict(delay_sum=sum(delays), delay_mean=float(np.mean(delays)), missed=missed, false=false)


def rule(T, need_car):
    return lambda e, a, b: (e['stop'][a:b] >= T) & (e['red'][a:b] == 0) & (e['fstop'][a:b] if need_car else True)


def gru_fixed(th):
    return lambda e, a, b: e['prob'][a:b] >= th


def gru_sched(hi, lo, t2):
    def f(e, a, b):
        t = e['stop'][a:b]
        th = np.where(t < 1.0, hi, hi + (lo - hi) * np.clip((t - 1.0) / max(t2 - 1.0, 1e-6), 0, 1))
        return e['prob'][a:b] >= th
    return f


P = {}
for T in [1, 2, 3, 5, 8, 10, 15, 20]:
    P[f'rule {T}s'] = ('rule', evaluate(rule(T, False)))
    P[f'rule {T}s+car'] = ('rule+car', evaluate(rule(T, True)))
for th in [0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995]:
    P[f'GRU P>={th}'] = ('gru', evaluate(gru_fixed(th)))
for hi in [0.95, 0.98, 0.99, 0.995]:
    for lo in [0.5, 0.7, 0.85]:
        for t2 in [5, 8, 12]:
            P[f'GRU {hi}->{lo} by {t2}s'] = ('sched', evaluate(gru_sched(hi, lo, t2)))

L = [f'Events: {n_pos} should-call, {n_neg} should-not-call  |  {n_min:.1f} min of episodes that never need a call', '',
     f'{"policy":28s} {"mean delay":>10s} {"missed":>6s} {"useless calls":>13s}']
for k, (fam, r) in P.items():
    if fam != 'sched':
        L.append(f'{k:28s} {r["delay_mean"]:9.1f}s {r["missed"]:6d} {r["false"]:8d}/{n_neg}')
L += ['', 'Best policy for a given call cost C (one useless LLM call = C seconds of being stuck):']
for C in [2, 5, 10, 20]:
    cost = {k: r['delay_sum'] + C * r['false'] for k, (fam, r) in P.items()}
    best = {}
    for k, (fam, r) in P.items():
        if fam not in best or cost[k] < cost[best[fam]]:
            best[fam] = k
    L.append(f'  C={C:>2}s: ' + ' | '.join(f'{fam}: {best[fam]} (cost {cost[best[fam]]:.0f})' for fam in ['rule', 'rule+car', 'gru', 'sched']))
rep = '\n'.join(L); print(rep)
open(os.path.join(OUT, f'tradeoff{TAG}.txt'), 'w', encoding='utf-8').write(rep + '\n')

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    sty = {'rule': ('s', '#7A8294', 'rule: stop >= T s'), 'rule+car': ('^', '#2F6FB0', 'rule: stop >= T s + stopped car'),
           'gru': ('o', '#D9661F', 'GRU fixed threshold'), 'sched': ('.', '#F2A65A', 'GRU time schedule')}
    for fam, (m, c, lab) in sty.items():
        pts = [(r['false'], r['delay_mean'], k) for k, (f, r) in P.items() if f == fam]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c, label=lab, s=50 if fam != 'sched' else 30, alpha=.85)
        if fam in ('rule', 'rule+car'):
            for x, y, k in pts:
                ax.annotate(k.split()[1], (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points', color=c)
    ax.set_xlabel(f'useless LLM calls (out of {n_neg} stops that resolve by themselves)')
    ax.set_ylabel('mean time until LLM is called (s)')
    ax.set_title('Call early vs. call uselessly  (lower-left is better)')
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'tradeoff{TAG}.png'), dpi=120)
    print('plot ->', os.path.join(OUT, f'tradeoff{TAG}.png'))
except Exception as ex:
    print('plot skipped:', ex)
