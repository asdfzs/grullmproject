"""
LLM advisor for the deadlock project: "the ego has been stopped behind something - WAIT or BYPASS?"

The call runs in a background thread so the driving loop never blocks; the agent keeps waiting
(PDM-Lite keeps control) until the answer arrives. Latency is measured and logged.

Modes (env LLM_MODE):
  stub    (default) no network. A few hand-written rules over the same summary text + a fake delay.
          Used to test the whole pipeline (trigger -> ask -> act) before paying for API calls.
  claude  Anthropic Messages API with the front-camera JPEG + the summary.
          needs env ANTHROPIC_API_KEY ; optional LLM_MODEL (default below), LLM_TIMEOUT (s)

Answer format (both modes): {"decision": "BYPASS"|"WAIT", "confidence": 0..1, "cues": [...], "reason": "..."}
"""
import os
import json
import time
import base64
import threading
import urllib.request

SYSTEM_PROMPT = """You are the high-level reasoning module of an autonomous car.
The low-level planner has stopped the car behind an object in its lane and has been waiting.
Decide the INTENT only:
- "WAIT"   : the blockage will clear by itself (traffic queue, car waiting at a light/junction, car about to move,
             pedestrians crossing, a vehicle that is just slow).
- "BYPASS" : the object will not move on its own (broken-down / stalled / parked / crashed vehicle, obstacle,
             hazard lights or warning triangle, nobody in the driver seat, long-stopped car with free road ahead of it).
The planner will only execute a bypass when the neighbouring lane is safe, so you do not judge gaps in traffic.
Use the camera image (front view of the ego car) and the numeric summary. If unsure, prefer "WAIT".
Reply with ONLY a JSON object:
{"decision": "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence"}"""


CORRIDOR_PROMPT = """You are the high-level reasoning module of an autonomous car.
Traffic cones block the car's lane. The map does not know about this construction zone, so the low-level planner
would simply follow its lane into the cones. The perception stack measured the free corridor between the cones.
Decide the INTENT only:
- "BYPASS": follow the temporary corridor (the planner shifts its path to the corridor centre and drives slowly).
            Only if the corridor is clearly wider than the car and no oncoming vehicle will use it at the same time.
- "WAIT"  : stop before the zone (e.g. an oncoming vehicle is in or near a single-lane corridor, or it is too narrow).
Use the camera image and the numeric summary. If unsure, choose WAIT.
Reply with ONLY a JSON object:
{"decision": "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence"}"""


DEBRIS_PROMPT = """You are the high-level reasoning module of an autonomous car.
An object is lying on the road in the car's lane ahead. The low-level planner did NOT recognise it as an obstacle.
Decide the INTENT only:
- "PROCEED": it is safe to drive over it (flat debris, cardboard, paper, a plastic bag, leaves, a thin mat).
- "BYPASS" : it must not be hit (box, barrel, tyre, rock, bin, cone, anything rigid or taller than about 15 cm).
- "WAIT"   : stop and wait (e.g. a person or animal next to it, or the lane is about to clear).
The summary also says how busy the lane a bypass would use is. A lane change into busy traffic is itself a risk:
if the object can be driven over, PROCEED instead of changing lanes. If it must not be hit, choose BYPASS anyway -
the planner waits for a safe gap before it changes lanes. If there is no lane to go around, choose WAIT.
Use the camera image and the numeric summary. If unsure between PROCEED and BYPASS, choose BYPASS.
Reply with ONLY a JSON object:
{"decision": "PROCEED" or "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence"}"""


def build_summary(row, extra=None):
    """Human-readable scene summary from one RecordingAgent row (dict)."""
    def g(k, default=0.0):
        try:
            return float(row.get(k, default))
        except Exception:
            return default
    lines = [
        f"ego stopped for {g('stop_duration_s'):.1f} s (speed {g('ego_speed_kmh'):.1f} km/h)",
        f"object ahead in ego lane: {row.get('front_class', 'none')}, gap {g('front_dist_m'):.1f} m, "
        f"its speed {g('front_speed_ms'):.1f} m/s, stopped for {g('front_stopped_duration_s'):.1f} s, "
        f"hazard lights {'ON' if g('front_hazard_on') > 0 else 'off'}",
        ("second object ahead in ego lane: none" if g('front2_dist_m') > 900 else
         f"second object ahead in ego lane: {g('front2_dist_m'):.1f} m away, speed {g('front2_speed_ms'):.1f} m/s"),
        f"left lane (same direction): {int(g('left_count'))} vehicles, mean speed {g('left_mean_speed_ms'):.1f} m/s",
        f"right lane (same direction): {int(g('right_count'))} vehicles, mean speed {g('right_mean_speed_ms'):.1f} m/s",
        ("traffic light ahead: none within 80 m" if g('traffic_light_dist_m') > 900 else
         f"traffic light ahead: {'RED/YELLOW' if g('traffic_light_red') > 0 else 'green'} at {g('traffic_light_dist_m'):.0f} m"),
        f"ego at junction: {'yes' if g('at_junction') > 0 else 'no'}",
    ]
    if extra:
        lines += extra
    return '\n'.join(lines)


def _parse_json(text):
    a, b = text.find('{'), text.rfind('}')
    d = json.loads(text[a:b + 1])
    dec = str(d.get('decision', '')).upper()
    d['decision'] = 'BYPASS' if dec.startswith('BYP') else ('PROCEED' if dec.startswith('PRO') else 'WAIT')
    d.setdefault('confidence', 0.5)
    d.setdefault('cues', [])
    d.setdefault('reason', '')
    return d


class LLMAdvisor:
    def __init__(self):
        self.mode = os.environ.get('LLM_MODE', 'stub').lower()
        self.model = os.environ.get('LLM_MODEL', 'claude-sonnet-5-5')
        self.timeout = float(os.environ.get('LLM_TIMEOUT', '20'))
        self.stub_delay = float(os.environ.get('LLM_STUB_DELAY', '1.5'))
        self.stub_bypass_after = float(os.environ.get('STUB_BYPASS_SEC', '30'))
        self._thread = None
        self._result = None
        self._t0 = None
        if self.mode == 'claude' and not os.environ.get('ANTHROPIC_API_KEY'):
            print('>> [LLMAdvisor] LLM_MODE=claude but ANTHROPIC_API_KEY is not set -> falling back to stub', flush=True)
            self.mode = 'stub'
        print(f'>> [LLMAdvisor] mode={self.mode}' + (f' model={self.model}' if self.mode == 'claude' else ''), flush=True)

    # ---------------------------------------------------------------- public
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def request(self, jpeg_bytes, summary, row):
        """Start one asynchronous query. Returns immediately."""
        self._result = None
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, args=(jpeg_bytes, summary, row), daemon=True)
        self._thread.start()

    def poll(self):
        """None while the answer is pending, else the answer dict (with 'latency_s')."""
        if self._thread is None or self._thread.is_alive():
            return None
        res, self._thread = self._result, None
        return res

    # ---------------------------------------------------------------- internals
    def _run(self, jpeg_bytes, summary, row):
        try:
            if self.mode == 'claude':
                res = self._ask_claude(jpeg_bytes, summary,
                                       {'debris': DEBRIS_PROMPT, 'corridor': CORRIDOR_PROMPT}.get(row.get('kind'), SYSTEM_PROMPT))
            else:
                res = self._ask_stub(row)
        except Exception as e:
            res = {'decision': 'WAIT', 'confidence': 0.0, 'cues': [], 'reason': f'LLM error -> default WAIT: {e}'}
        res['latency_s'] = round(time.time() - self._t0, 2)
        res['mode'] = self.mode
        self._result = res

    def _ask_stub(self, row):
        """Stand-in for the LLM. Uses only cues a camera could see; NOT the scenario name."""
        time.sleep(self.stub_delay)
        if row.get('kind') == 'corridor':
            w = float(row.get('corr_width', 0.0))
            ew = float(row.get('ego_width', 2.0))
            n = int(row.get('oncoming_n', 0))
            d = float(row.get('oncoming_dist', 999.0))
            cues = [f'corridor {w:.1f} m (car {ew:.1f} m)', f'{n} oncoming' + (f' at {d:.0f} m' if n else '')]
            if w < ew + 0.8:
                return {'decision': 'WAIT', 'confidence': 0.6, 'cues': cues,
                        'reason': 'stub: corridor too narrow for the car - stop before the cones'}
            if n and d < 60.0:
                return {'decision': 'WAIT', 'confidence': 0.6, 'cues': cues,
                        'reason': 'stub: single-lane corridor and an oncoming vehicle - let it pass first'}
            return {'decision': 'BYPASS', 'confidence': 0.7, 'cues': cues,
                    'reason': 'stub: cones on both lane edges leave a temporary corridor on the centre line - follow it slowly'}
        if row.get('kind') == 'debris':
            h = float(row.get('obj_h', 1.0))
            name = str(row.get('obj_type', '')).replace('static.prop.', '')
            busy = bool(row.get('adj_busy', False))
            cues = [f'{name}, {h:.2f} m high', f"bypass lane {'busy' if busy else 'free'} ({row.get('adj_count', 0)} cars)"]
            if h <= 0.20:
                return {'decision': 'PROCEED', 'confidence': 0.7, 'cues': cues,
                        'reason': 'stub: low/flat object, can be driven over' +
                                  (' - no need to risk a lane change into busy traffic' if busy else '')}
            if not row.get('adj_possible', True):
                return {'decision': 'WAIT', 'confidence': 0.6, 'cues': cues,
                        'reason': 'stub: object must not be hit and there is no lane to go around'}
            return {'decision': 'BYPASS', 'confidence': 0.7, 'cues': cues,
                    'reason': 'stub: object taller than 20 cm, must not be hit' +
                              (' - lane is busy, planner must wait for a safe gap' if busy else '')}
        hazard = float(row.get('front_hazard_on', 0)) > 0
        fstop = float(row.get('front_stopped_duration_s', 0))
        f2_moving = float(row.get('front2_dist_m', 999)) < 900 and float(row.get('front2_speed_ms', 0)) > 0.5
        if hazard:
            return {'decision': 'BYPASS', 'confidence': 0.9, 'cues': ['hazard lights'],
                    'reason': 'stub: front vehicle shows hazard lights -> likely broken down'}
        if fstop >= self.stub_bypass_after and not f2_moving:
            return {'decision': 'BYPASS', 'confidence': 0.6, 'cues': [f'stopped {fstop:.0f}s'],
                    'reason': f'stub: front vehicle has not moved for {fstop:.0f}s'}
        return {'decision': 'WAIT', 'confidence': 0.6, 'cues': [],
                'reason': 'stub: no evidence the front vehicle is disabled yet'}

    def _ask_claude(self, jpeg_bytes, summary, system_prompt=SYSTEM_PROMPT):
        content = []
        if jpeg_bytes:
            content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                        'data': base64.b64encode(jpeg_bytes).decode('ascii')}})
        content.append({'type': 'text', 'text': 'Scene summary from the perception stack:\n' + summary})
        body = json.dumps({'model': self.model, 'max_tokens': 300, 'system': system_prompt,
                           'messages': [{'role': 'user', 'content': content}]}).encode('utf-8')
        req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=body, method='POST', headers={
            'x-api-key': os.environ['ANTHROPIC_API_KEY'],
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode('utf-8'))
        text = ''.join(c.get('text', '') for c in data.get('content', []) if c.get('type') == 'text')
        res = _parse_json(text)
        res['raw'] = text
        return res
