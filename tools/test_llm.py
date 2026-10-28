"""
Check that the Claude API key works BEFORE running CARLA (costs well under 1 cent).
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    venv\Scripts\python.exe tools\test_llm.py
It sends the same kind of question the car will send (summary text; plus a camera JPEG if one exists).
"""
import os, sys, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'team_code'))
os.environ.setdefault('LLM_MODE', 'claude')
from llm_advisor import LLMAdvisor, build_summary

if not os.environ.get('ANTHROPIC_API_KEY'):
    sys.exit('ANTHROPIC_API_KEY is not set. In PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."')

row = {'stop_duration_s': 12, 'ego_speed_kmh': 0, 'front_class': 'vehicle_stopped', 'front_dist_m': 3.8,
       'front_speed_ms': 0, 'front_stopped_duration_s': 17, 'front_hazard_on': 0, 'front2_dist_m': 999,
       'front2_speed_ms': 0, 'left_count': 2, 'left_mean_speed_ms': 9, 'right_count': 0, 'right_mean_speed_ms': 0,
       'traffic_light_dist_m': 999, 'traffic_light_red': 0, 'at_junction': 0}
jpgs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'results', 'intervention', '*', '*.jpg')))
jpeg = open(jpgs[-1], 'rb').read() if jpgs else b''
print('image:', jpgs[-1] if jpgs else 'none (text only)')

a = LLMAdvisor()
summary = build_summary(row)
print('--- question ---\n' + summary)
a.request(jpeg, summary, row)
while a.busy():
    time.sleep(0.1)
res = a.poll()
print('--- answer ---')
for k, v in res.items():
    print(f'{k}: {v}')
if 'error' in str(res.get('reason', '')).lower():
    print('\n!! The call failed. Common causes: wrong key, no credit on the account, or LLM_MODEL name not available.')
