"""
LLM advisor for the deadlock project: "the ego has been stopped behind something - WAIT or BYPASS?"

The call runs in a background thread so the driving loop never blocks; the agent keeps waiting
(PDM-Lite keeps control) until the answer arrives. Latency is measured and logged.

Modes (env LLM_MODE):
  stub    (default) no network. A few hand-written rules over the same summary text + a fake delay.
          Used to test the whole pipeline (trigger -> ask -> act) before paying for API calls.
  claude  Anthropic Messages API with the front-camera JPEG + the summary.
          needs env ANTHROPIC_API_KEY ; optional LLM_MODEL (default below), LLM_TIMEOUT (s)
  gemini  Google Gemini API (free tier works) with the front-camera JPEG + the summary.
          needs env GEMINI_API_KEY ; optional LLM_MODEL (default gemini-3.5-flash), LLM_TIMEOUT (s)
If the real LLM fails (network, quota 429, bad JSON) the stub rules answer instead and 'reason' says so.

Answer format (both modes): {"decision": "BYPASS"|"WAIT", "confidence": 0..1, "cues": [...], "reason": "..."}
"""
import os
import json
import time
import base64
import threading
import urllib.request
import urllib.error

SYSTEM_PROMPT = """You are the high-level reasoning module of an autonomous car.
The low-level planner has stopped the car behind an object in its lane and has been waiting.
Decide the INTENT only:
- "WAIT"   : the blockage will clear by itself (traffic queue, car waiting at a light/junction, car about to move,
             pedestrians crossing, a vehicle that is just slow).
- "BYPASS" : the object will not move on its own (broken-down / stalled / parked / crashed vehicle, obstacle,
             hazard lights or warning triangle, nobody in the driver seat, long-stopped car with free road ahead of it).
The planner will only execute a bypass when the neighbouring lane is safe, so you do not judge gaps in traffic.
Important: a long stop by itself is NOT evidence. Queues on a busy road often stand still for 20-40 s.
Overtaking a queue is a wrong and risky move, while WAIT is cheap: you will be asked again about every 10 s,
with the updated stop time. So choose "BYPASS" only with a positive cue that the vehicle is disabled, e.g.
hazard lights, a warning triangle, an open door or bonnet, a crash, OR the stopped vehicle has free road ahead of it
(no vehicle in front of it) while it does not move, OR it has been stopped for more than about 40 s.
The strongest cue without hazard lights: the road ahead of the stopped vehicle is FREE, there is no red light or
junction that explains the stop, and it still does not move (especially if traffic in the neighbouring lane keeps
passing it, or it was already standing before we arrived) -> "BYPASS" once the ego has waited about 10 s.
If another stopped vehicle stands right in front of it and there is no such cue, it is most likely a queue -> "WAIT".
Common sense on a road with another lane in OUR direction: if that lane is free or flowing, there is no red light or
junction ahead, and our lane has stood still for about 20 s, changing lanes around the stopped vehicle(s) is normal
driving even if it might be a queue -> "BYPASS" (the planner only changes lanes when the gap is safe). Do not do this
when the only lane to go around is the ONCOMING lane - there keep waiting for a clear cue.
Use the camera image (front view of the ego car) and the numeric summary. If unsure, prefer "WAIT".
When the photo and the numbers disagree: trust the numbers for distances, speeds and durations, and the photo for
what kind of object it is. Never invent things that are neither in the photo nor in the summary.
Reply with ONLY a JSON object:
{"decision": "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence",
 "seen_in_image": "one sentence: what you actually see in the photo (or 'no image')"}"""


CORRIDOR_PROMPT = """You are the high-level reasoning module of an autonomous car.
Traffic cones stand in or along the car's lane. The map does not know about this construction zone, so the
low-level planner cannot plan through it by itself. The perception stack measured the free corridor between the cones.
Decide the INTENT only:
- "BYPASS": follow the temporary corridor (the planner shifts its path to the corridor centre and drives slowly).
            Only if the corridor is clearly wider than the car and no oncoming vehicle will use it at the same time.
- "WAIT"  : stop before the zone (e.g. an oncoming vehicle is in or near a single-lane corridor, or it is too narrow).
Use the camera image and the numeric summary. If unsure, choose WAIT.
Also say WHERE the way through is, seen from our car: "path": "left" (across the lane marking on the left of our
lane), "center" (inside our own lane, straight ahead), "right" (across the lane marking on the right of our lane),
or "none" (no way through).
When the photo and the numbers disagree: trust the numbers for distances, speeds and durations, and the photo for
what kind of object it is. Never invent things that are neither in the photo nor in the summary.
Reply with ONLY a JSON object:
{"decision": "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence",
 "seen_in_image": "one sentence: what you actually see in the photo (or 'no image')",
 "path": "left" or "center" or "right" or "none"}"""


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
When the photo and the numbers disagree: trust the numbers for distances, speeds and durations, and the photo for
what kind of object it is. Never invent things that are neither in the photo nor in the summary.
Reply with ONLY a JSON object:
{"decision": "PROCEED" or "BYPASS" or "WAIT", "confidence": number 0..1, "cues": [short strings], "reason": "one sentence",
 "seen_in_image": "one sentence: what you actually see in the photo (or 'no image')"}"""


# ---------------------------------------------------------------- "plain" prompts (LLM_INPUT=plain / photo)
# Same roles and answer format, but NO hints that already contain the answer: no thresholds (15 cm, 40 s),
# no list of example objects, no "free road ahead -> BYPASS" rule. Used to measure what the LLM itself understands.
_FMT = """When the photo and the numbers disagree: trust the numbers for distances, speeds and durations, and the photo for
what kind of object it is. Never invent things that are neither in the photo nor in the summary.
Reply with ONLY a JSON object:
{"decision": %s, "confidence": number 0..1, "cues": [short strings], "reason": "one sentence",
 "seen_in_image": "one sentence: what you actually see in the photo (or 'no image')"}"""

SYSTEM_PROMPT_PLAIN = """You are the high-level reasoning module of an autonomous car.
The low-level planner has stopped the car behind a vehicle in its lane and has been waiting.
Decide the INTENT only:
- "WAIT"   : the blockage will clear by itself, so keep waiting behind it.
- "BYPASS" : the vehicle will not move on its own, so the planner should go around it.
The planner only changes lanes when the neighbouring lane is safe. You will be asked again about every 10 s.
Use the camera image (front view of the ego car) and the numeric summary.
""" + _FMT % '"BYPASS" or "WAIT"'

DEBRIS_PROMPT_PLAIN = """You are the high-level reasoning module of an autonomous car.
An object is lying on the road in the car's lane ahead. The low-level planner did NOT recognise it as an obstacle.
Decide the INTENT only:
- "PROCEED": drive over it.
- "BYPASS" : do not hit it; the planner changes lanes around it (only when the neighbouring lane is safe).
- "WAIT"   : stop in front of it and wait.
Use the camera image and the numeric summary.
""" + _FMT % '"PROCEED" or "BYPASS" or "WAIT"'

CORRIDOR_PROMPT_PLAIN = """You are the high-level reasoning module of an autonomous car.
Traffic cones stand in or along the car's lane; this construction zone is not in the map. Decide the INTENT only:
- "BYPASS": drive through the zone slowly along the way through (the planner shifts its path there).
- "WAIT"  : stop before the zone.
Use the camera image and the numeric summary.
Also say WHERE the way through is, seen from our car: "path": "left" (across the lane marking on the left of our
lane), "center" (inside our own lane, straight ahead), "right" (across the lane marking on the right of our lane),
or "none" (no way through).
""" + (_FMT % '"BYPASS" or "WAIT"')[:-1] + ',\n "path": "left" or "center" or "right" or "none"}'


def input_level():
    """LLM_INPUT: full (default: hints + interpreted cue line) | plain (raw numbers, no hints) | photo (plain, and
    the debris size is not given - the object must be judged from the photo)."""
    v = os.environ.get('LLM_INPUT', 'full').lower()
    return v if v in ('full', 'plain', 'photo') else 'full'


def _side_line(row, side, g):
    kind = row.get(f'{side}_lane_kind')
    if kind == 'none':
        return f"{side} lane: there is NO {side} lane (road edge / no driving lane)"
    if kind == 'opposite':
        return f"{side} lane: oncoming-traffic lane (opposite direction)"
    return (f"{side} lane (same direction): {int(g(f'{side}_count'))} vehicles, "
            f"mean speed {g(f'{side}_mean_speed_ms'):.1f} m/s")


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
        _side_line(row, 'left', g),
        _side_line(row, 'right', g),
        ("traffic light ahead: none within 80 m" if g('traffic_light_dist_m') > 900 else
         f"traffic light ahead: {'RED/YELLOW' if g('traffic_light_red') > 0 else 'green'} at {g('traffic_light_dist_m'):.0f} m"),
        f"ego at junction: {'yes' if g('at_junction') > 0 else 'no'}",
    ]
    # interpreted cues (what a human driver would notice) - only for a stopped vehicle ahead, only for LLM_INPUT=full
    if row.get('front_class') == 'vehicle_stopped' and input_level() == 'full':
        cues = []
        if g('front2_dist_m') > 900:
            cues.append("the road ahead of the stopped vehicle is FREE (no vehicle within ~50 m), yet it does not move")
        elif g('front2_speed_ms') > 1.0:
            cues.append("the vehicle in front of it is driving away, yet it does not move")
        else:
            cues.append("another vehicle is standing right in front of it (looks like a queue)")
        side_flow = [(n, g(f'{n}_mean_speed_ms')) for n in ('left', 'right') if g(f'{n}_count') > 0]
        moving = [f"{n} lane ({v:.1f} m/s)" for n, v in side_flow if v > 3.0]
        if moving:
            cues.append("traffic in the neighbouring lane keeps passing it: " + ', '.join(moving))
        elif side_flow:
            cues.append("the neighbouring lane is also (almost) stopped -> general congestion")
        if g('front_stopped_duration_s') - g('stop_duration_s') > 3.0:
            cues.append(f"it was already standing still before we arrived "
                        f"({g('front_stopped_duration_s') - g('stop_duration_s'):.0f} s before us)")
        if g('traffic_light_dist_m') < 900 and g('traffic_light_red') > 0:
            cues.append("there is a red light ahead that explains the stop")
        lines.append("cues: " + '; '.join(cues))
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


# ---------------------------------------------------------------- safety guard
# The LLM proposes, these fixed rules have the last word. Every change is written into the answer
# ('guard' = why, 'llm_decision' = what the LLM said) so the logs show when a rule overrode the LLM.
GUARD_MAX_PROCEED_H = 0.35   # m: never drive over something estimated taller than this (PROCEED props <= 0.21 m)
GUARD_MIN_STOP_S = 10.0      # s: never overtake a vehicle without hazard lights that has stood still less than this
GUARD_MIN_CONF = 0.6         # below this an intent that changes the car's behaviour is not trusted


def guard(res, row):
    kind = row.get('kind', 'vehicle')
    dec, conf = res.get('decision', 'WAIT'), float(res.get('confidence', 0.5) or 0.0)
    new, why = dec, None

    def f(k, d):
        try:
            return float(row.get(k, d))
        except Exception:
            return d
    if kind == 'debris':
        h = f('obj_h', 1.0) if row.get('obj_h') is not None else None
        if dec == 'PROCEED' and h is not None and h > GUARD_MAX_PROCEED_H:
            new, why = 'BYPASS', f'estimated height {h:.2f} m > {GUARD_MAX_PROCEED_H} m: never drive over it'
        elif dec == 'PROCEED' and conf < GUARD_MIN_CONF:
            new, why = 'BYPASS', f'PROCEED with low confidence {conf:.2f}: if unsure, do not hit it'
        if new == 'BYPASS' and not row.get('adj_possible', True):
            new, why = 'WAIT', (why + '; ' if why else '') + 'no lane to go around'
    elif kind == 'corridor':
        if dec == 'PROCEED':
            new, why = 'WAIT', 'PROCEED is not a valid corridor intent'
        elif dec == 'BYPASS':
            w, ew = f('corr_width', 0.0), f('ego_width', 2.0)
            n, d = int(f('oncoming_n', 0)), f('oncoming_dist', 999.0)
            if w < ew + 0.6:
                new, why = 'WAIT', f'corridor {w:.1f} m too narrow for the car ({ew:.1f} m + 0.6 m margin)'
            elif n and d < 40.0:
                new, why = 'WAIT', f'oncoming vehicle {d:.0f} m away'
            elif conf < GUARD_MIN_CONF:
                new, why = 'WAIT', f'BYPASS with low confidence {conf:.2f}'
            elif isinstance(row.get('corr_options'), dict) and row.get('corr_options'):
                opts = row['corr_options']           # every passable corridor, by side
                said = str(res.get('path', '')).lower().strip()
                res['path_measured'] = '/'.join(sorted(opts))
                if said in ('left', 'center', 'right') and said not in opts:
                    new, why = 'WAIT', (f'LLM wants to go {said}, but there is no passable corridor on the {said} '
                                        f'(passable: {res["path_measured"]})')
                elif said not in ('left', 'center', 'right'):
                    if len(opts) > 1:
                        new, why = 'WAIT', f'several corridors ({res["path_measured"]}) and the LLM gave no side'
                    else:
                        res['path_note'] = f'LLM gave no usable path ({said or "missing"}) -> the only corridor used'
            else:
                c = f('corr_center', 0.0)            # measured corridor centre: - left / + right of our lane centre
                measured = 'left' if c < -0.6 else ('right' if c > 0.6 else 'center')
                said = str(res.get('path', '')).lower().strip()
                res['path_measured'] = measured
                if said in ('left', 'center', 'right') and said != measured:
                    new, why = 'WAIT', (f'LLM sees the way through on the {said}, but the measured corridor is on '
                                        f'the {measured}: do not drive until both agree')
                elif said not in ('left', 'center', 'right'):
                    res['path_note'] = f'LLM gave no usable path ({said or "missing"}) -> measured corridor used'
    else:   # stopped vehicle ahead
        if dec == 'PROCEED':
            new, why = 'WAIT', 'PROCEED is not a valid intent behind a stopped vehicle'
        elif dec == 'BYPASS':
            if f('traffic_light_red', 0) > 0:
                new, why = 'WAIT', 'red light ahead explains the stop'
            elif f('front2_dist_m', 999) < 900 and f('front2_speed_ms', 0) > 1.0:
                new, why = 'WAIT', 'the vehicle in front of it is driving away: the queue is releasing'
            elif f('front_hazard_on', 0) <= 0 and f('front_stopped_duration_s', 0) < GUARD_MIN_STOP_S:
                new, why = 'WAIT', (f"no hazard lights and stopped only {f('front_stopped_duration_s', 0):.1f} s "
                                    f"(< {GUARD_MIN_STOP_S:.0f} s): too early to call it disabled")
            elif conf < GUARD_MIN_CONF and f('front_hazard_on', 0) <= 0:
                new, why = 'WAIT', f'BYPASS with low confidence {conf:.2f} and no hazard lights'
    if new != dec:
        res['llm_decision'] = dec
        res['decision'] = new
        res['guard'] = why
    return res


class LLMAdvisor:
    def __init__(self):
        self.mode = os.environ.get('LLM_MODE', 'stub').lower()
        self.model = os.environ.get('LLM_MODEL') or ('gemini-3-flash-preview' if self.mode == 'gemini' else 'claude-sonnet-5-5')
        self.timeout = float(os.environ.get('LLM_TIMEOUT', '20'))
        self.stub_delay = float(os.environ.get('LLM_STUB_DELAY', '1.5'))
        self.stub_bypass_after = float(os.environ.get('STUB_BYPASS_SEC', '30'))
        self._thread = None
        self._result = None
        self._t0 = None
        if self.mode == 'claude' and not os.environ.get('ANTHROPIC_API_KEY'):
            print('>> [LLMAdvisor] LLM_MODE=claude but ANTHROPIC_API_KEY is not set -> falling back to stub', flush=True)
            self.mode = 'stub'
        if self.mode == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
            print('>> [LLMAdvisor] LLM_MODE=gemini but GEMINI_API_KEY is not set -> falling back to stub', flush=True)
            self.mode = 'stub'
        print(f'>> [LLMAdvisor] mode={self.mode}' + (f' model={self.model}' if self.mode in ('claude', 'gemini') else ''), flush=True)

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
        if input_level() == 'full':
            prompt = {'debris': DEBRIS_PROMPT, 'corridor': CORRIDOR_PROMPT}.get(row.get('kind'), SYSTEM_PROMPT)
        else:
            prompt = {'debris': DEBRIS_PROMPT_PLAIN, 'corridor': CORRIDOR_PROMPT_PLAIN}.get(row.get('kind'),
                                                                                           SYSTEM_PROMPT_PLAIN)
        used = self.mode
        try:
            if self.mode == 'claude':
                res = self._ask_claude(jpeg_bytes, summary, prompt)
            elif self.mode == 'gemini':
                res = self._ask_gemini(jpeg_bytes, summary, prompt)
            else:
                res = self._ask_stub(row)
        except Exception as e:
            err = str(e)
            if hasattr(e, 'read'):
                try:
                    err += ' | ' + e.read().decode('utf-8', 'replace')[:300]
                except Exception:
                    pass
            print(f'>> [LLMAdvisor] {self.mode} call failed -> stub answers instead: {err}', flush=True)
            try:
                self.stub_delay, d = 0.0, self.stub_delay
                res = self._ask_stub(row)
                self.stub_delay = d
                res['reason'] = f'LLM error ({err[:120]}) -> stub fallback: ' + res.get('reason', '')
            except Exception:
                res = {'decision': 'WAIT', 'confidence': 0.0, 'cues': [], 'reason': f'LLM error -> default WAIT: {err}'}
            used = 'stub_fallback'
        res = guard(res, row)
        res['latency_s'] = round(time.time() - self._t0, 2)
        res['mode'] = used
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
            c = float(row.get('corr_center', 0.0))
            return {'decision': 'BYPASS', 'confidence': 0.7, 'cues': cues,
                    'path': 'left' if c < -0.6 else ('right' if c > 0.6 else 'center'),
                    'reason': 'stub: cones on both lane edges leave a temporary corridor on the centre line - follow it slowly'}
        if row.get('kind') == 'debris':
            h = float(row.get('obj_h') if row.get('obj_h') is not None else 1.0)
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

    def _ask_gemini(self, jpeg_bytes, summary, system_prompt=SYSTEM_PROMPT):
        parts = []
        if jpeg_bytes:
            parts.append({'inline_data': {'mime_type': 'image/jpeg',
                                          'data': base64.b64encode(jpeg_bytes).decode('ascii')}})
        parts.append({'text': 'Scene summary from the perception stack:\n' + summary})
        body = json.dumps({
            'system_instruction': {'parts': [{'text': system_prompt}]},
            'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'temperature': 0.0, 'maxOutputTokens': 2048,
                                 'responseMimeType': 'application/json'},
        }).encode('utf-8')
        # free-tier models are sometimes "overloaded" (503) or out of quota (429): try a 2nd model before giving up
        # default: the SAME model once more after a short pause (keep one model for every decision);
        # set LLM_MODEL_FALLBACK to another model name to use that one instead
        fallback = os.environ.get('LLM_MODEL_FALLBACK', '') or self.model
        models = [self.model, fallback]
        for i, model in enumerate(models):
            if i > 0:
                time.sleep(2.0)
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            req = urllib.request.Request(url, data=body, method='POST', headers={
                'x-goog-api-key': os.environ['GEMINI_API_KEY'],
                'content-type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode('utf-8'))
                self.used_model = model
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and i + 1 < len(models):
                    print(f'>> [LLMAdvisor] {model} HTTP {e.code} -> trying {models[i + 1]}', flush=True)
                    continue
                raise
        cands = data.get('candidates') or []
        if not cands:
            raise RuntimeError(f'no answer: {json.dumps(data)[:200]}')
        text = ''.join(p.get('text', '') for p in cands[0].get('content', {}).get('parts', [])
                       if not p.get('thought'))
        res = _parse_json(text)
        res['raw'] = text
        res['model'] = self.used_model
        return res
