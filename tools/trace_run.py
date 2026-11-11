"""
One run, step by step: what went INTO the LLM, what came OUT, what the car did, and whether it drove on.
    venv\Scripts\python.exe tools\trace_run.py                 (latest run)
    venv\Scripts\python.exe tools\trace_run.py 165_stalled     (latest run whose folder starts with this)
Writes the same text to <run folder>\trace.txt  (the photos sent to the LLM are in that folder too).
"""
import os, sys, json, glob, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'team_code'))
import llm_advisor as L

pat = sys.argv[1] if len(sys.argv) > 1 else ''
runs = sorted(glob.glob(os.path.join(ROOT, 'results', 'intervention', pat + '*')), key=os.path.getmtime)
if not runs:
    sys.exit('no run found')
run = runs[-1]
name = os.path.basename(run)
evs = [json.loads(l) for l in open(os.path.join(run, 'events.jsonl'), encoding='utf-8')]
csvp = os.path.join(ROOT, 'results', 'intervention_features', name + '.csv')
rows = list(csv.DictReader(open(csvp, encoding='utf-8'))) if os.path.exists(csvp) else []
out = []
def say(s=''):
    print(s); out.append(s)

def speed_at(t):
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(float(r['sim_time_s']) - t))
    return float(r['ego_speed_kmh'])

say(f'RUN {name}')
start = next((e for e in evs if e['event'] == 'start'), {})
say(f"trigger mode={start.get('trigger')}  llm={start.get('llm')}  llm_input={start.get('llm_input', 'full')}"
    f"  perception={start.get('perception', '?')}")
for e in evs:
    ev = e['event']
    if ev == 'trigger':
        kind = e.get('kind', 'vehicle')
        prompt = {'debris': 'DEBRIS_PROMPT', 'corridor': 'CORRIDOR_PROMPT'}.get(kind, 'SYSTEM_PROMPT (stopped vehicle)')
        say('\n' + '=' * 90)
        say(f"[t={e['t']:.1f}s] LLM CALL {e.get('call')}  fired by: {e.get('by')}"
            + (f"  (GRU p={e.get('p_gru')}, stopped {e.get('stop_s')} s)" if kind == 'vehicle' else '')
            + f"  ego speed {speed_at(e['t'])} km/h")
        say(f"  photo sent : {e.get('image') or 'NONE (text only)'}")
        say(f"  instructions: {prompt}")
        say('  text sent  :')
        for line in str(e.get('summary', '')).splitlines():
            say('     | ' + line)
    elif ev == 'llm_answer':
        say(f"[t={e['t']:.1f}s] LLM ANSWER after {e.get('latency_s')} s ({e.get('mode')}, {e.get('model', '')})")
        say(f"  decision   : {e.get('decision')}  (confidence {e.get('confidence')})")
        if e.get('guard'):
            say(f"  !! GUARD   : LLM said {e.get('llm_decision')} -> changed to {e.get('decision')}: {e.get('guard')}")
        say(f"  cues       : {e.get('cues')}")
        say(f"  reason     : {e.get('reason')}")
        say(f"  in photo   : {e.get('seen_in_image', '-')}")
        if e.get('path') or e.get('path_measured'):
            say(f"  path       : LLM says {e.get('path', '-')} / measured {e.get('path_measured', '-')}"
                + (f"  ({e.get('path_note')})" if e.get('path_note') else ''))
        if e.get('raw'):
            say(f"  raw output : {str(e.get('raw'))[:600]}")
    elif ev in ('start', 'tick'):
        continue
    else:
        extra = {k: v for k, v in e.items() if k not in ('t', 'event', 'summary', 'raw')}
        say(f"[t={e['t']:.1f}s] CONTROL/EVENT {ev}: {json.dumps(extra, ensure_ascii=False)[:300]}")

if rows:
    say('\n' + '=' * 90)
    say('speed timeline (every 2 s): t / km/h / stop_s / front object')
    last = -9
    for r in rows:
        t = float(r['sim_time_s'])
        if t - last >= 2.0:
            last = t
            say(f"  {t:5.1f}s  {float(r['ego_speed_kmh']):5.1f} km/h  stop {float(r['stop_duration_s']):5.1f}s  "
                f"front {r['front_class']} {float(r['front_dist_m']):.0f}m")
    ans = [e for e in evs if e['event'] == 'llm_answer']
    if ans:
        t_ans = ans[-1]['t']
        after = [float(r['ego_speed_kmh']) for r in rows if float(r['sim_time_s']) > t_ans]
        say(f"\nafter the last answer: max speed {max(after) if after else 0:.1f} km/h"
            f" -> {'drove on normally' if after and max(after) > 15 else 'did NOT drive on'}")
end = next((e for e in evs if e['event'] == 'end'), None)
say(f"end: {json.dumps(end, ensure_ascii=False) if end else 'no end event (crash or timeout?)'}")
open(os.path.join(run, 'trace.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\nsaved', os.path.join(run, 'trace.txt'))
