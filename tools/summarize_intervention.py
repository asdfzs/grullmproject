"""Print one line per intervention run: route result + what the trigger/LLM did."""
import sys, os, json, glob

W = os.environ.get('WORK_DIR', r'C:\carla_garage')
for tag in sys.argv[1:]:
    res = os.path.join(W, 'results', 'intervention_logs', tag + '.json')
    status, comp, infr = '?', '?', []
    try:
        d = json.load(open(res, encoding='utf-8'))
        rec = d['_checkpoint']['records'][-1]
        status = rec.get('status', '?')
        comp = rec.get('scores', {}).get('score_route', '?')
        infr = [k for k, v in rec.get('infractions', {}).items() if v]
    except Exception as e:
        status = f'no result ({e.__class__.__name__})'
    print(f'{tag:40s} | {status} | route completion {comp} | infractions {infr or "none"}')
# events of the newest intervention folders
for d in sorted(glob.glob(os.path.join(W, 'results', 'intervention', '*')), key=os.path.getmtime)[-len(sys.argv[1:]):]:
    print('  --', os.path.basename(d))
    try:
        for line in open(os.path.join(d, 'events.jsonl'), encoding='utf-8'):
            e = json.loads(line)
            if e['event'] == 'trigger':
                print(f"     t={e['t']:6.1f}s TRIGGER  stop={e['stop_s']}s p_gru={e.get('p_gru')}")
            elif e['event'] == 'llm_answer':
                print(f"     t={e['t']:6.1f}s LLM -> {e['decision']} (conf {e.get('confidence')}, {e.get('latency_s')}s) {e.get('reason','')[:90]}")
            elif e['event'] in ('handed_to_planner', 'backup_start', 'backup_done', 'bypass_impossible'):
                print(f"     t={e['t']:6.1f}s {e['event']}")
    except Exception as ex:
        print('     (no events)', ex)
