r"""
Gemini API check BEFORE running CARLA (free tier: no cost).
    $env:GEMINI_API_KEY = "<your key>"
    venv\Scripts\python.exe tools\test_gemini.py
1) checks the key and lists the Gemini models this key can use
2) photo-only test: the saved prop photos (10 m) are sent WITHOUT the object name or size,
   so Gemini must decide PROCEED / BYPASS from the image alone
3) one stopped-car question (camera JPEG + summary), like the GRU trigger sends
Result table is also saved to results\llm_test\gemini_test_<time>.txt
"""
import os, sys, glob, json, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'team_code'))
os.environ['LLM_MODE'] = 'gemini'
from llm_advisor import LLMAdvisor, build_summary

KEY = (os.environ.get('GEMINI_API_KEY') or '').strip()
if KEY:
    os.environ['GEMINI_API_KEY'] = KEY
if not KEY:
    sys.exit('GEMINI_API_KEY is empty (or only a space/newline). In PowerShell, on ONE line:  $env:GEMINI_API_KEY = "<your key>"')

out = []
def say(s=''):
    print(s, flush=True); out.append(s)

# ---- 1) key + models
try:
    req = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models?pageSize=200',
                                 headers={'x-goog-api-key': KEY})
    models = json.loads(urllib.request.urlopen(req, timeout=20).read().decode('utf-8')).get('models', [])
except Exception as e:
    body = e.read().decode('utf-8', 'replace')[:300] if hasattr(e, 'read') else ''
    sys.exit(f'!! key check failed: {e} {body}')
flash = [m['name'].split('/')[-1] for m in models
         if 'generateContent' in m.get('supportedGenerationMethods', []) and 'flash' in m['name']]
say('key OK. flash models: ' + ', '.join(flash[:15]))

a = LLMAdvisor()
say(f'using model: {a.model}  (change with $env:LLM_MODEL = "<name from the list>")')

def ask(jpeg, summary, row):
    a.request(jpeg, summary, row)
    while a.busy():
        time.sleep(0.1)
    return a.poll()

# ---- 2) photo-only prop test
EXPECT = {'ironplank': 'PROCEED', 'creasedbox03': 'PROCEED', 'creasedbox01': 'PROCEED', 'dirtdebris01': 'PROCEED',
          'box03': 'BYPASS', 'barrel': 'BYPASS', 'trashcan01': 'BYPASS', 'trafficcone01': 'BYPASS',
          'constructioncone': 'BYPASS'}
say('\n== photo-only test (no name, no size given) ==')
say(f"{'prop':18s} {'expected':9s} {'gemini':8s} {'conf':5s} {'sec':5s} reason")
ok = n = 0
for prop, exp in EXPECT.items():
    dirs = sorted(glob.glob(os.path.join(ROOT, 'results', 'intervention', f'165_prop_{prop}_*')))
    photo = next((os.path.join(d, 'photo_d10.jpg') for d in reversed(dirs)
                  if os.path.exists(os.path.join(d, 'photo_d10.jpg'))), None)
    if not photo:
        say(f'{prop:18s} (no photo_d10.jpg found, skipped)'); continue
    summary = ("object lying on the road in the ego lane: unknown object (a static object, not a vehicle), "
               "size not measured, about 10 m ahead\n"
               "left lane (the one a bypass would use): 0 vehicles nearby -> free\n")
    row = {'kind': 'debris', 'obj_type': 'unknown', 'obj_h': 1.0, 'adj_busy': False, 'adj_count': 0}
    res = ask(open(photo, 'rb').read(), summary, row)
    n += 1; ok += res['decision'] == exp
    mark = 'OK ' if res['decision'] == exp else 'XX '
    say(f"{prop:18s} {exp:9s} {mark}{res['decision']:5s} {float(res.get('confidence', 0)):.2f}  "
        f"{res['latency_s']:4.1f}  [{res.get('mode')}] {res.get('reason', '')[:110]}")
    time.sleep(4)          # stay well under the free-tier per-minute limit
say(f'photo-only correct: {ok}/{n}')

# ---- 3) stopped-car question
say('\n== stopped car (GRU-trigger style) ==')
jpgs = [p for p in sorted(glob.glob(os.path.join(ROOT, 'results', 'intervention', '*stalled*', 'call*.jpg')))]
jpeg = open(jpgs[-1], 'rb').read() if jpgs else b''
row = {'stop_duration_s': 12, 'ego_speed_kmh': 0, 'front_class': 'vehicle_stopped', 'front_dist_m': 3.8,
       'front_speed_ms': 0, 'front_stopped_duration_s': 17, 'front_hazard_on': 0, 'front2_dist_m': 999,
       'front2_speed_ms': 0, 'left_count': 0, 'left_mean_speed_ms': 0, 'right_count': 0, 'right_mean_speed_ms': 0,
       'traffic_light_dist_m': 999, 'traffic_light_red': 0, 'at_junction': 0}
res = ask(jpeg, build_summary(row), row)
say('image: ' + (os.path.relpath(jpgs[-1], ROOT) if jpgs else 'none'))
for k in ('decision', 'confidence', 'cues', 'reason', 'latency_s', 'mode'):
    say(f'  {k}: {res.get(k)}')
if any('fallback' in str(r) for r in out):
    say('\n!! some calls fell back to the stub. Check the error text above (wrong key, 429 quota, model name).')

os.makedirs(os.path.join(ROOT, 'results', 'llm_test'), exist_ok=True)
fn = os.path.join(ROOT, 'results', 'llm_test', time.strftime('gemini_test_%Y%m%d_%H%M%S.txt'))
open(fn, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\nsaved:', fn)
