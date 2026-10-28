"""
Debris comparison: PDM-Lite alone vs 'always bypass' vs our LLM decision, with/without a busy neighbour lane.

    venv\\Scripts\\python.exe tools\\debris_report.py _db_pdm _db_always _db_llm

For every log results\\intervention_logs\\<route>_<trigger>_<llm><suffix>_run<r>.json (route contains 'debris'):
  result / completion / collisions      from the leaderboard record
  decision, lane change, stop time,     from results\\intervention\\<route>_<trigger><suffix>_run<r>_*\\events.jsonl
  closest car in the other lane         (min_gap_other_lane_m: <5 m = risky merge)
  stop time (whole route)               from results\\intervention_features\\<same tag>_*.csv
"""
import os, sys, glob, json, csv, re
from collections import defaultdict

W = os.environ.get('WORK_DIR', r'C:\carla_garage')
suffixes = sys.argv[1:] or ['_db_pdm', '_db_always', '_db_llm']
NAMES = {'_db_pdm': 'PDM-Lite only', '_db_always': 'always bypass', '_db_llm': 'ours (LLM)'}


def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


rows = []
for suf in suffixes:
    for f in sorted(glob.glob(os.path.join(W, 'results', 'intervention_logs', f'*debris*{suf}_run*.json'))):
        tag = os.path.basename(f)[:-5]
        m = re.match(rf'(.+?)_(rule|gru)_(stub|claude){re.escape(suf)}_run(\d+)$', tag)
        if not m:
            continue
        rid, trig, _, run = m.groups()
        try:
            rec = json.load(open(f, encoding='utf-8'))['_checkpoint']['records'][-1]
            status, comp = rec['status'], rec['scores']['score_route']
            infr = {k: len(v) for k, v in rec['infractions'].items() if v and k != 'min_speed_infractions'}
            dur = rec['meta'].get('duration_game') or 0
        except Exception:
            status, comp, infr, dur = 'NO RESULT', 0, {}, 0
        coll = sum(v for k, v in infr.items() if 'collision' in k)
        ev = latest(os.path.join(W, 'results', 'intervention', f'{rid}_{trig}{suf}_run{run}_*', 'events.jsonl'))
        dec, passed, adj = '-', {}, {}
        if ev:
            for line in open(ev, encoding='utf-8'):
                e = json.loads(line)
                if e['event'] == 'llm_answer' and e.get('kind') == 'debris':
                    dec = e['decision']
                elif e['event'] == 'trigger' and e.get('by') == 'always_bypass':
                    dec = 'BYPASS(rule)'
                elif e['event'] == 'debris_seen' and not adj:
                    adj = e
                elif e['event'] == 'debris_passed' and not passed:
                    passed = e
        stop_total = None
        cf = latest(os.path.join(W, 'results', 'intervention_features', f'{rid}_{trig}{suf}_run{run}_*.csv'))
        if cf:
            with open(cf, encoding='utf-8') as fh:
                sp = [float(r['ego_speed_kmh']) for r in csv.DictReader(fh)]
            stop_total = sum(1 for v in sp if v < 1.0) * 0.05
        rows.append(dict(route=rid, policy=NAMES.get(suf, suf), run=run, status=status, comp=comp, dur=dur, coll=coll,
                         infr=infr, dec=dec, seen=adj.get('gap_m'), busy=adj.get('adj_busy'),
                         lane=passed.get('lane_changed'), stop_ep=passed.get('stop_s'),
                         gap=passed.get('min_gap_other_lane_m'), stop_total=stop_total,
                         mwait=passed.get('merge_wait_s'), fbrake=passed.get('max_forced_brake_ms2')))

print(f'{"route":24s} {"policy":14s} {"run":3s} {"result":9s} {"compl":>5s} {"time":>6s} {"coll":>4s} {"seen@":>6s} '
      f'{"busy":5s} {"decision":12s} {"lanechg":7s} {"stop@obj":>8s} {"stop all":>8s} {"min gap":>7s} {"merge wait":>10s} {"brake forced":>12s}')
def fmt(v, unit, nd=1, skip=None):
    if v is None or (skip is not None and v >= skip):
        return '-'
    return f'{v:.{nd}f}{unit}'


for r in rows:
    print(f'{r["route"]:24s} {r["policy"]:14s} {r["run"]:3s} {r["status"][:9]:9s} {r["comp"]:5.0f} {r["dur"]:6.1f} {r["coll"]:4d} '
          f'{fmt(r["seen"], "m", 0):>6s} {str(r["busy"])[:5]:5s} {r["dec"][:12]:12s} {str(r["lane"])[:7]:7s} '
          f'{fmt(r["stop_ep"], "s"):>8s} {fmt(r["stop_total"], "s"):>8s} {fmt(r["gap"], "m", 1, 999):>7s} {fmt(r["mwait"], "s"):>10s} {fmt(r["fbrake"], "m/s2"):>12s}')

print('\n==== average per route x policy ====')
agg = defaultdict(list)
for r in rows:
    agg[(r['route'], r['policy'])].append(r)
print(f'{"route":24s} {"policy":14s} {"n":>2s} {"completed":>9s} {"collisions":>10s} {"lane changes":>12s} {"stop s":>7s} {"min gap":>8s}')
for (route, pol), rs in sorted(agg.items()):
    n = len(rs)
    done = sum(r['comp'] >= 99 for r in rs)
    coll = sum(r['coll'] for r in rs)
    lc = sum(bool(r['lane']) for r in rs)
    st = [r['stop_total'] for r in rs if r['stop_total'] is not None]
    gs = [r['gap'] for r in rs if r['gap'] is not None and r['gap'] < 999]
    print(f'{route:24s} {pol:14s} {n:2d} {done:>6d}/{n:<2d} {coll:10d} {lc:>9d}/{n:<2d} '
          f'{(sum(st) / len(st) if st else float("nan")):7.1f} {(f"{min(gs):.1f}m" if gs else "-"):>8s}')
