"""
What PDM-Lite does when it drives straight into each prop (RoadDebris NOT registered, our debris logic OFF).

    venv\\Scripts\\python.exe tools\\prop_drive_report.py            (reads the *_prop runs)

Per prop: collision with the object (leaderboard infraction), route result, lowest speed after reaching
the object, longest standstill. 'passed over' = no collision and no long stop.
"""
import os, glob, json, csv, re

W = os.environ.get('WORK_DIR', r'C:\carla_garage')
SUF = '_prop'
rows = []
for f in sorted(glob.glob(os.path.join(W, 'results', 'intervention_logs', f'165_prop_*_gru_stub{SUF}_run*.json'))):
    tag = os.path.basename(f)[:-5]
    prop = re.match(r'165_prop_(.+?)_gru_stub', tag).group(1)
    try:
        rec = json.load(open(f, encoding='utf-8'))['_checkpoint']['records'][-1]
        status, comp = rec['status'], rec['scores']['score_route']
        infr = {k: v for k, v in rec['infractions'].items() if v and k != 'min_speed_infractions'}
        dur = rec['meta'].get('duration_game') or 0
    except Exception:
        status, comp, infr, dur = 'NO RESULT', 0, {}, 0
    coll = sum(len(v) for k, v in infr.items() if 'collision' in k)
    # our own collision log (written even when the run ends early on MAX_STOP)
    evs = sorted(glob.glob(os.path.join(W, 'results', 'intervention', f'165_prop_{prop}_gru{SUF}_run*_*', 'events.jsonl')),
                 key=os.path.getmtime)
    hit = None
    if evs:
        for line in open(evs[-1], encoding='utf-8'):
            e = json.loads(line)
            if e.get('event') == 'collision' and str(e.get('other', '')).startswith('static.prop'):
                if hit is None or e['impulse'] > hit['impulse']:
                    hit = e
    if hit is not None:
        coll = max(coll, 1)
    blocked = 'agent_blocked' in ''.join(infr.keys()) or any('blocked' in k for k in infr)
    csvs = sorted(glob.glob(os.path.join(W, 'results', 'intervention_features', f'165_prop_{prop}_gru{SUF}_run*_*.csv')),
                  key=os.path.getmtime)
    vmin, stop_max = None, 0.0
    if csvs:
        rs = list(csv.DictReader(open(csvs[-1], encoding='utf-8')))
        sp = [float(r['ego_speed_kmh']) for r in rs]
        st = [float(r['stop_duration_s']) for r in rs]
        moving = [i for i, v in enumerate(sp) if v > 20]
        if moving:
            after = sp[moving[0]:]
            vmin = min(after)
            stop_max = max(st[moving[0]:])
    if coll and (stop_max > 10 or comp < 99):
        verdict = 'hit it and got stuck'
    elif coll:
        verdict = 'touched it (collision) but drove on'
    elif stop_max > 10:
        verdict = 'stopped in front (no collision)'
    else:
        verdict = 'drove over / past it'
    hit_txt = '-' if hit is None else f"{hit['impulse']:.0f} @ {hit['speed_kmh']:.0f} km/h"
    rows.append((prop, status, comp, round(dur, 1), coll, vmin, round(stop_max, 1), verdict, hit_txt))

print(f'{"prop":18s} {"result":10s} {"compl":>5s} {"time":>6s} {"coll":>4s} {"hit (impulse @ speed)":>22s} {"min km/h":>8s} {"stop s":>6s}  verdict')
for p, st, c, d, co, vm, sm, v, ht in rows:
    print(f'{p:18s} {st[:10]:10s} {c:5.0f} {d:6.1f} {co:4d} {ht:>22s} {("-" if vm is None else f"{vm:.1f}"):>8s} {sm:6.1f}  {v}')
