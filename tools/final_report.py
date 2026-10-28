"""
Final test summary: full system (GRU+fallback trigger -> LLM -> planner) on situations it was never built on.

    venv\\Scripts\\python.exe tools\\final_report.py _final

Reads results\\intervention_logs\\<route>_gru_stub<suffix>_run*.json and the matching results\\intervention\\<route>_gru_*\\events.jsonl
Expected behaviour per route name:
  *stalled / *broken  -> deadlock: must call the LLM and bypass, then finish the route
  *queue / *stopgo    -> resolves by itself: ideally NO call; a call answered WAIT is harmless, a BYPASS is a mistake
  official routes     -> normal traffic situations PDM-Lite already handles: ideally no call, same completion
"""
import os, sys, glob, json

W = os.environ.get('WORK_DIR', r'C:\carla_garage')
suffix = sys.argv[1] if len(sys.argv) > 1 else ''
rows = []
for f in sorted(glob.glob(os.path.join(W, 'results', 'intervention_logs', f'*_gru_stub{suffix}_run*.json'))):
    tag = os.path.basename(f)[:-5]
    rid = tag.split('_gru_stub')[0]
    try:
        rec = json.load(open(f, encoding='utf-8'))['_checkpoint']['records'][-1]
        status, comp = rec['status'], rec['scores']['score_route']
        infr = {k: len(v) for k, v in rec['infractions'].items() if v and k != 'min_speed_infractions'}
        dur = rec['meta'].get('duration_game')
    except Exception:
        status, comp, infr, dur = 'NO RESULT', 0, {}, None
    folders = sorted(glob.glob(os.path.join(W, 'results', 'intervention', f'{rid}_gru_*')), key=os.path.getmtime)
    calls, answers, bypass, first = [], [], False, None
    if folders:
        for line in open(os.path.join(folders[-1], 'events.jsonl'), encoding='utf-8'):
            e = json.loads(line)
            if e['event'] == 'trigger':
                calls.append(e.get('by', '?')); first = first if first is not None else e.get('stop_s')
            elif e['event'] == 'llm_answer':
                answers.append(e['decision'])
            elif e['event'] == 'handed_to_planner':
                bypass = True
    if 'debris' in rid:
        kind = 'debris_pass' if any(k in rid for k in ('small', 'plank')) else 'debris_hit'
    else:
        kind = 'deadlock' if any(k in rid for k in ('stalled', 'broken')) else ('resolves' if any(k in rid for k in ('queue', 'stopgo')) else 'official')
    if kind == 'debris_hit':            # rigid object: must go around, no collision
        ok = bypass and comp >= 99 and not infr
    elif kind == 'debris_pass':         # flat object: drive over, no bypass needed
        ok = not bypass and comp >= 99 and not infr
    elif kind == 'deadlock':
        ok = bypass and comp >= 99 and not infr
    elif kind == 'resolves':
        ok = not bypass and comp >= 99 and not infr
    else:
        ok = not bypass and not infr
    rows.append((rid, kind[:9], status, comp, dur, infr, calls, answers, bypass, first, ok))

print(f'{"route":18s} {"kind":9s} {"result":10s} {"compl":>5s} {"time":>6s}  {"LLM calls (by)":28s} {"answers":22s} {"bypass":6s} {"1st call @stop":>14s} {"infractions":18s} OK?')
for rid, kind, status, comp, dur, infr, calls, answers, bypass, first, ok in rows:
    print(f'{rid:18s} {kind:9s} {status[:10]:10s} {comp:5.0f} {(dur or 0):6.1f}  {str(len(calls))+" "+str(calls)[:24]:28s} {",".join(answers)[:22]:22s} {str(bypass):6s} '
          f'{("-" if first is None else f"{first:.1f}s"):>14s} {str(infr)[:18]:18s} {"YES" if ok else "NO"}')
n_ok = sum(r[-1] for r in rows)
useless = sum(len(r[6]) for r in rows if r[1] in ('resolves', 'official'))
print(f'\n{n_ok}/{len(rows)} routes behaved as expected | LLM calls where none was needed: {useless}')
