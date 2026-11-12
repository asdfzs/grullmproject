"""
Export every REAL LLM call from the CARLA runs as a training set for a small "student" model (distillation).
No CARLA needed. Same core keys as Taeyoung's distill_data.jsonl, plus what the simulator knows.
    venv\\Scripts\\python.exe tools\\export_distill.py                (-> distill_export\\)
    venv\\Scripts\\python.exe tools\\export_distill.py --all-teachers (also stub answers, marked teacher='stub')

One line in samples.jsonl = one LLM call:
  feature_vec / feature_names  34 GRU features of the sensor row at that moment (gru_features.featurize)
  window                       the last 20 s of those features at 4 Hz (80 x 34) - same input the GRU sees
  row                          the raw sensor row (CSV) at that moment
  summary / image              exactly the text and the photo the LLM got (images\\<file>)
  llm_*                        what the LLM answered (decision, confidence, cues, reason, seen_in_image, path, model)
  guard / final_decision       whether a safety rule changed it, and what the car actually did
  kind                         vehicle | debris | corridor
  scenario / truth             scenario of the run and what a correct driver does there (for EVALUATION, not a label)
"""
import os, sys, json, glob, csv, shutil, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'gru'))
from gru_features import featurize, FEATURE_NAMES

TRUTH = {  # scenario keyword -> what a correct driver ends up doing (for evaluation)
    'broken': 'BYPASS', 'stalled': 'BYPASS', 'queue': 'WAIT', 'shortqueue': 'WAIT', 'stopgo': 'WAIT',
    'ironplank': 'PROCEED', 'creasedbox': 'PROCEED', 'dirtdebris01': 'PROCEED', 'debris_small': 'PROCEED',
    'box03': 'BYPASS', 'barrel': 'BYPASS', 'trashcan': 'BYPASS', 'debris_large': 'BYPASS',
    'closure_left': 'BYPASS/left', 'closure_right': 'BYPASS/right', 'corridor_closed': 'WAIT/none',
    'corridor_lane': 'BYPASS/center', 'arrow_left': 'BYPASS/left', 'arrow_right': 'BYPASS/left',
    'sign_': 'BYPASS/left', 'r174_corridor': 'BYPASS/left',
}


def scenario_of(run):
    for k in sorted(TRUTH, key=len, reverse=True):
        if k in run:
            return k, TRUTH[k]
    return None, None


ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'distill_export'))
ap.add_argument('--all-teachers', action='store_true', help='also keep stub / stub_fallback answers')
a = ap.parse_args()
os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)

out, n_img, by_kind = [], 0, {}
for ev_path in sorted(glob.glob(os.path.join(ROOT, 'results', 'intervention', '*', 'events.jsonl'))):
    run_dir = os.path.dirname(ev_path)
    run = os.path.basename(run_dir)
    csv_path = os.path.join(ROOT, 'results', 'intervention_features', run + '.csv')
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8'))) if os.path.exists(csv_path) else []
    evs = [json.loads(l) for l in open(ev_path, encoding='utf-8') if l.strip()]
    start = next((e for e in evs if e['event'] == 'start'), {})
    scen, truth = scenario_of(run)
    pending = []   # triggers waiting for their answer (answers come in order)
    for i, e in enumerate(evs):
        if e['event'] == 'trigger' and e.get('by') != 'always_bypass':
            pending.append(e)
            continue
        if e['event'] != 'llm_answer' or not pending:
            continue
        tr = pending.pop(0)
        teacher = e.get('mode', 'stub')
        if teacher != 'gemini' and teacher != 'claude' and not a.all_teachers:
            continue
        kind = tr.get('kind', 'vehicle')
        row, window = None, None
        if rows:
            k = min(range(len(rows)), key=lambda j: abs(float(rows[j]['sim_time_s']) - tr['t']))
            row = rows[k]
            hist = [rows[j] for j in range(max(0, k - 395), k + 1, 5)]       # 20 s at 4 Hz
            hist = [hist[0]] * (80 - len(hist)) + hist                         # pad at the start
            window = [[round(v, 4) for v in featurize(r)] for r in hist[-80:]]
        img = tr.get('image')
        img_out = None
        if img and os.path.exists(os.path.join(run_dir, img)):
            img_out = f'{run}__{img}'
            shutil.copyfile(os.path.join(run_dir, img), os.path.join(a.out, 'images', img_out))
            n_img += 1
        # what the car did after this answer
        final = e.get('decision')
        for f in evs[i + 1:]:
            if f['event'] in ('trigger', 'llm_answer'):
                break
            if f['event'] in ('bypass_skipped', 'bypass_aborted'):
                final = 'WAIT'
            if f['event'] == 'route_mismatch':
                final = 'WAIT'
        s = {
            'sample_id': len(out), 'run': run, 't': tr['t'], 'kind': kind, 'call': tr.get('call'),
            'fired_by': tr.get('by'), 'gru_p': tr.get('p_gru'), 'llm_input': start.get('llm_input', 'full'),
            'perception': start.get('perception'),
            'row': row, 'feature_names': FEATURE_NAMES,
            'feature_vec': [round(v, 4) for v in featurize(row)] if row else None,
            'window': window,
            'summary': tr.get('summary'), 'image': img_out,
            'llm_decision': e.get('llm_decision', e.get('decision')),     # what the LLM said (before any rule)
            'llm_confidence': e.get('confidence'), 'llm_cues': e.get('cues'), 'llm_reason': e.get('reason'),
            'llm_seen_in_image': e.get('seen_in_image'), 'llm_path': e.get('path'),
            'llm_model': e.get('model'), 'teacher': teacher, 'latency_s': e.get('latency_s'),
            'guard': e.get('guard'), 'discarded': bool(e.get('discarded')), 'final_decision': final,
            'scenario': scen, 'truth': truth,
            'debris': ({x: tr.get(x) for x in ('front', 'gap_m', 'obj_h_true', 'obj_h_est')} if kind == 'debris' else None),
            'corridor': ({x: tr.get(x) for x in ('n_cones', 'gap_m', 'corridor_width_m', 'corridor_center_m', 'oncoming')}
                         if kind == 'corridor' else None),
        }
        out.append(s)
        by_kind[kind] = by_kind.get(kind, 0) + 1

with open(os.path.join(a.out, 'samples.jsonl'), 'w', encoding='utf-8') as f:
    for s in out:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
# small overview table
with open(os.path.join(a.out, 'overview.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sample_id', 'run', 'kind', 't', 'llm_decision', 'llm_path', 'final_decision', 'truth', 'guard',
                'discarded', 'llm_model', 'teacher', 'image'])
    for s in out:
        w.writerow([s['sample_id'], s['run'], s['kind'], s['t'], s['llm_decision'], s['llm_path'], s['final_decision'],
                    s['truth'], (s['guard'] or '')[:60], s['discarded'], s['llm_model'], s['teacher'], s['image']])
print(f'{len(out)} samples ({by_kind}), {n_img} images -> {a.out}')
