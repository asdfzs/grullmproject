"""
InterventionAgent = PDM-Lite + feature logging (RecordingAgent) + deadlock trigger + LLM advisor.

Flow every tick (20 Hz):
  1. PDM-Lite drives as usual (it stops behind the unregistered broken/stalled car or the queue).
  2. Trigger decides whether this stop is worth asking the LLM about:
        TRIGGER_MODE=rule : stop >= TRIGGER_STOP_SEC, no red light, stopped vehicle within 15 m
        TRIGGER_MODE=gru  : online GRU probability >= GRU_THRESH (model from GRU_DIR) + stopped vehicle ahead
  3. LLM advisor (async; stub or Claude) answers WAIT or BYPASS from the front camera + summary.
        WAIT   -> keep waiting, ask again after REASK_SEC (max MAX_LLM_CALLS calls)
        BYPASS -> (optionally back up to BACKUP_GAP m first) then hand the obstacle to PDM-Lite as
                  'ParkedObstacle' in CarlaDataProvider.active_scenarios. PDM-Lite then plans the lane
                  shift itself and still applies its own collision checks (the LLM never steers).

Output per episode: results/intervention/<tag>_<time>/events.jsonl + the camera JPEG sent at each call.

Env (set by run_intervene.ps1): TRIGGER_MODE, TRIGGER_STOP_SEC (10), GRU_DIR, GRU_THRESH (0.85),
  REASK_SEC (10), MAX_LLM_CALLS (5), BACKUP_GAP (7; 0 = never back up), LLM_MODE (stub|claude), ...
"""
import os
import json
import math
import time
import datetime

import carla
import numpy as np

from recording_agent import RecordingAgent, DT
from llm_advisor import LLMAdvisor, build_summary
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
  return 'InterventionAgent'


class InterventionAgent(RecordingAgent):

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager)
    env = os.environ.get
    self._trig_mode = env('TRIGGER_MODE', 'rule').lower()
    self._trig_stop = float(env('TRIGGER_STOP_SEC', '10'))
    self._reask = float(env('REASK_SEC', '10'))
    self._max_calls = int(env('MAX_LLM_CALLS', '5'))
    self._backup_gap = float(env('BACKUP_GAP', '7'))
    # safety-net rule used together with the GRU: call anyway after this long behind a stopped car (0 = off)
    self._fallback = float(env('FALLBACK_STOP_SEC', '0'))
    self._fired_by = None
    # fallen objects (static props) lying in the ego lane - PDM-Lite does not see them at all
    self._debris_on = env('DEBRIS', '1') == '1'
    self._debris_range = float(env('DEBRIS_RANGE', '60'))
    self._debris_state = None
    self._debris_done = set()
    # temporary corridor through a construction zone (cones), not in the map
    self._corr = None
    self._corr_done = set()
    # PERCEPTION=perfect : simulator truth (exact size, name, seen from 60 m)
    # PERCEPTION=noisy   : camera-like - seen only from a size-dependent distance and after 0.25 s of continuous
    #                      sight, size/distance estimated with noise, object class unknown to the LLM
    self._perception = env('PERCEPTION', 'perfect').lower()
    # DEBRIS_POLICY=llm           : ask the LLM (PROCEED / BYPASS / WAIT)
    # DEBRIS_POLICY=always_bypass : baseline - every detected object is bypassed, no LLM
    self._debris_policy = env('DEBRIS_POLICY', 'llm').lower()
    # SAFE_MERGE=1: PDM-Lite does NOT check cars coming from behind in the target lane for a same-direction
    # bypass (it just shifts its route). So we hold the car until the target lane has a real gap, then hand over.
    self._safe_merge = env('SAFE_MERGE', '0') == '1'
    self._merge_headway = float(env('MERGE_HEADWAY_S', '1.5'))
    self._seen_ticks = {}
    import random as _r
    self._prng = _r.Random()
    self._gru = None
    self._advisor = LLMAdvisor()
    if self._trig_mode == 'gru':
      import sys
      root = env('WORK_DIR', '.')
      sys.path.insert(0, os.path.join(root, 'gru'))
      from gru_features import OnlineGRU
      gdir = env('GRU_DIR', os.path.join(root, 'gru', 'out_long'))
      self._gru = OnlineGRU(gdir)
      self._gru_thresh = float(env('GRU_THRESH', '0.85'))
      print(f'>> [Intervention] GRU trigger from {gdir} (label={self._gru.label}, thresh={self._gru_thresh}, '
            f'history {self._gru.win}x{self._gru.step} ticks, fallback={self._fallback}s)', flush=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self._ev_dir = os.path.join(env('WORK_DIR', '.'), 'results', 'intervention', f'{self._rec_tag}_{stamp}')
    os.makedirs(self._ev_dir, exist_ok=True)
    self._ev_file = open(os.path.join(self._ev_dir, 'events.jsonl'), 'w', encoding='utf-8')

    self._calls = 0
    self._last_ask_t = -1e9
    self._pending_actor = None
    self._pending_t = None
    self._handled = set()          # actor ids already handed to PDM-Lite
    self._backing_up = None        # (actor, direction, t_start) while reversing
    self._use_cam = env('LLM_CAMERA', '1' if self._advisor.mode == 'claude' else '0') == '1'
    self._snap_cam = None
    self._snap_img = None
    self._snap_wait = None      # (actor, t_started, row) while waiting for the camera frame
    self._p_gru = 0.0
    self._log('start', trigger=self._trig_mode, llm=self._advisor.mode, trig_stop=self._trig_stop)
    print(f'>> [Intervention] trigger={self._trig_mode} llm={self._advisor.mode} -> {self._ev_dir}', flush=True)

  # ------------------------------------------------------------------ camera (on demand)
  # NOTE: adding the camera to sensors() froze the Windows CARLA build at sensor spawn.
  # So the camera is spawned by us only when the LLM is asked, grabs one frame, and is destroyed.
  def _start_snapshot(self):
    world = CarlaDataProvider.get_world()
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '960')
    bp.set_attribute('image_size_y', '540')
    bp.set_attribute('fov', '90')
    tf = carla.Transform(carla.Location(x=0.8, z=1.6), carla.Rotation(pitch=-3.0))
    self._snap_img = None
    self._snap_cam = world.spawn_actor(bp, tf, attach_to=self._vehicle)

    def cb(image):
      if self._snap_img is None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        self._snap_img = arr[:, :, :3].copy()
    self._snap_cam.listen(cb)

  def _stop_snapshot(self):
    try:
      if self._snap_cam is not None:
        self._snap_cam.stop()
        self._snap_cam.destroy()
    except Exception:
      pass
    self._snap_cam = None

  # ------------------------------------------------------------------ helpers
  def _t(self):
    return self._rec_tick * DT

  def _log(self, event, **kw):
    rec = {'t': round(self._t(), 2), 'event': event}
    rec.update(kw)
    self._ev_file.write(json.dumps(rec, ensure_ascii=False) + '\n')
    self._ev_file.flush()
    print(f'>> [Intervention] t={rec["t"]:.1f}s {event} ' + json.dumps(kw, ensure_ascii=False)[:300], flush=True)

  def _jpeg(self):
    if self._snap_img is None:
      return b''
    try:
      import cv2
      ok, buf = cv2.imencode('.jpg', self._snap_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
      return buf.tobytes() if ok else b''
    except Exception:
      return b''

  def _bypass_direction(self):
    """Which side the obstacle is relative to the free lane, in PDM-Lite's convention:
    'right' = obstacle is on the right -> shift LEFT. Returns (scenario_type, direction) or (None, None)."""
    m = CarlaDataProvider.get_map()
    wp = m.get_waypoint(self._vehicle.get_location())
    drive = carla.LaneType.Driving
    left, right = wp.get_left_lane(), wp.get_right_lane()
    if left is not None and left.lane_type == drive and left.lane_id * wp.lane_id > 0:
      return 'ParkedObstacle', 'right'
    if right is not None and right.lane_type == drive and right.lane_id * wp.lane_id > 0:
      return 'ParkedObstacle', 'left'
    if left is not None and left.lane_type == drive:       # opposite-direction lane -> two-way overtake
      return 'ParkedObstacleTwoWays', 'right'
    return None, None

  def _stopped_chain_end(self, actor, max_gap=15.0):
    """Last car of a row of stopped cars starting at `actor` in the ego lane (e.g. a stalled car with
    another parked car in front of it). PDM-Lite then shifts the route around the WHOLE row instead of
    merging back right after the first car (which left the ego stuck beside the second one)."""
    ego_tf = self._vehicle.get_transform()
    cands = []
    for a in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
      if a.id == self._vehicle.id:
        continue
      lon, lat = self._ego_frame(ego_tf, a.get_location())
      v = a.get_velocity()
      if abs(lat) < 1.8 and lon > 0 and (v.x * v.x + v.y * v.y) ** 0.5 < 0.3:
        cands.append((lon, a))
    cands.sort(key=lambda e: e[0])
    last = actor
    last_lon = self._ego_frame(ego_tf, actor.get_location())[0]
    for lon, a in cands:
      if lon <= last_lon:
        continue
      if lon - last_lon > max_gap:
        break
      last, last_lon = a, lon
    return last

  def _hand_to_planner(self, actor, direction_info):
    stype, direction = direction_info
    last = self._stopped_chain_end(actor)
    # same data layout the official ParkedObstacle / ParkedObstacleTwoWays scenarios register
    data = [actor, None if last.id == actor.id else last, direction, False, 1e9, 1e9, False]
    if last.id != actor.id:
      self._log('obstacle_row', first=actor.id, last=last.id, last_type=last.type_id)
    CarlaDataProvider.active_scenarios.append((stype, data))
    self._handled.add(actor.id)
    self._log('handed_to_planner', actor=actor.type_id, actor_id=actor.id, scenario_type=stype, direction=direction)

  def _rear_clear(self, need=8.0):
    ego = self._vehicle
    tf = ego.get_transform()
    for a in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
      if a.id == ego.id:
        continue
      lon, lat = self._ego_frame(tf, a.get_location())
      if -need - 5.0 < lon < 0 and abs(lat) < 1.8:
        return False
    return True

  def _trigger(self, row):
    near_stopped = row['front_class'] == 'vehicle_stopped' and row['front_dist_m'] < 15.0
    no_red = row['traffic_light_red'] == 0
    if self._gru is not None:
      self._p_gru = self._gru.push(row)
      if near_stopped and self._p_gru >= self._gru_thresh:
        self._fired_by = 'gru'
        return True
      if self._fallback > 0 and near_stopped and no_red and row['stop_duration_s'] >= self._fallback:
        self._fired_by = f'fallback>={self._fallback:g}s'
        return True
      return False
    self._fired_by = 'rule'
    return near_stopped and row['stop_duration_s'] >= self._trig_stop and no_red

  # ------------------------------------------------------------------ main
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)
    try:
      control = self._intervene(control)
    except Exception as e:  # never break driving because of the intervention layer
      print(f'>> [Intervention] error: {e}', flush=True)
    return control

  def _intervene(self, control):
    row = self._last_row
    if row is None:
      return control
    t = self._t()
    fire = self._trigger(row)     # evaluated every tick so the GRU sees a continuous stream

    # ---- executing a back-up manoeuvre before the bypass
    if self._backing_up is not None:
      actor, dinfo, t0 = self._backing_up
      gap = row['front_dist_m']
      if (not actor.is_alive) or gap >= self._backup_gap or t - t0 > 6.0 or not self._rear_clear(3.0):
        self._backing_up = None
        self._log('backup_done', gap_m=gap, took_s=round(t - t0, 1))
        if actor.is_alive:
          self._hand_to_planner(actor, dinfo)
        return control
      return carla.VehicleControl(throttle=0.35, steer=0.0, brake=0.0, reverse=True)

    # ---- waiting for the one camera frame (max 2 s sim time)
    if self._snap_wait is not None:
      actor, t0, kind = self._snap_wait
      if self._snap_img is None and t - t0 < 2.0:
        return self._debris_brake(control) if kind == 'debris' else control
      self._stop_snapshot()
      self._snap_wait = None
      if kind == 'debris':
        self._ask_debris(actor, row, t)
        return self._debris_brake(control)
      self._ask(actor, row, t)
      return control

    # ---- construction zone: cones leave only a temporary corridor that is not in the map
    if self._debris_on:
      out = self._corridor_step(control, row, t)
      if out is not None:
        return out

    # ---- fallen object in the lane
    if self._debris_on:
      out = self._debris_step(control, row, t)
      if out is not None:
        return out

    # ---- waiting for an LLM answer
    if self._pending_actor is not None:
      res = self._advisor.poll()
      if res is None:
        return control
      actor = self._pending_actor
      self._pending_actor = None
      self._last_ask_t = t
      self._log('llm_answer', sim_wait_s=round(t - self._pending_t, 2), **res)
      if res['decision'] == 'BYPASS' and actor.is_alive:
        dinfo = self._bypass_direction()
        if dinfo[0] is None:
          self._log('bypass_impossible', reason='no adjacent driving lane')
          self._handled.add(actor.id)
          return control
        if self._backup_gap > 0 and row['front_dist_m'] < self._backup_gap and self._rear_clear():
          self._backing_up = (actor, dinfo, t)
          self._log('backup_start', gap_m=row['front_dist_m'], target_gap_m=self._backup_gap)
        else:
          self._hand_to_planner(actor, dinfo)
      return control

    # ---- trigger
    if not fire:
      return control
    actor = self._last_front_actor
    if actor is None or actor.id in self._handled:
      return control
    if t - self._last_ask_t < self._reask or self._calls >= self._max_calls:
      return control

    self._calls += 1
    if self._use_cam:
      try:
        self._start_snapshot()
        self._snap_wait = (actor, t, 'vehicle')
        return control
      except Exception as e:
        print(f'>> [Intervention] camera spawn failed ({e}) -> text only', flush=True)
    self._ask(actor, row, t)
    return control

  def _ask(self, actor, row, t):
    jpeg = self._jpeg()
    img_name = f'call{self._calls}_t{t:.1f}.jpg'
    if jpeg:
      with open(os.path.join(self._ev_dir, img_name), 'wb') as f:
        f.write(jpeg)
    summary = build_summary(row)
    self._log('trigger', call=self._calls, by=self._fired_by, stop_s=row['stop_duration_s'], p_gru=round(self._p_gru, 3),
              front=actor.type_id, front_id=actor.id, image=img_name if jpeg else None, summary=summary)
    self._pending_actor = actor
    self._pending_t = t
    self._advisor.request(jpeg, summary, dict(row))

  # ------------------------------------------------------------------ fallen objects
  def _scan_debris(self):
    """Closest static prop lying in the ego lane within DEBRIS_RANGE m ahead -> (gap_m, lat_m, actor) or None."""
    world = CarlaDataProvider.get_world()
    cmap = CarlaDataProvider.get_map()
    tf = self._vehicle.get_transform()
    ewp = cmap.get_waypoint(tf.location)
    best = None
    for a in world.get_actors().filter('static.prop.*'):
      if a.id in self._debris_done or 'cone' in a.type_id:
        continue                         # cones are handled as a construction zone (_corridor_step)
      lon, lat = self._ego_frame(tf, a.get_location())
      rng = self._debris_range
      if self._perception == 'noisy':
        h = 2 * a.bounding_box.extent.z
        rng = min(50.0, 25.0 + 40.0 * h)    # small/flat things are noticed late (camera resolution)
      if not (0.0 < lon < rng):
        self._seen_ticks.pop(a.id, None)
        continue
      awp = cmap.get_waypoint(a.get_location())
      if awp is None or awp.lane_id * ewp.lane_id <= 0:
        continue
      # in the ego lane? the object's own lane must be the lane we will drive into (follow the lane, works on
      # curves), and the object must sit near that lane's centre line
      if awp.transform.location.distance(a.get_location()) > 1.6 + a.bounding_box.extent.y:
        continue
      if not self._lane_ahead_contains(ewp, awp, lon + 10.0):
        continue
      gap = lon - a.bounding_box.extent.x - self._vehicle.bounding_box.extent.x
      if self._perception == 'noisy':
        self._seen_ticks[a.id] = self._seen_ticks.get(a.id, 0) + 1
        if self._seen_ticks[a.id] < 5:      # must be seen for 0.25 s before it counts as a detection
          continue
      if best is None or gap < best[0]:
        best = (gap, lat, a)
    return best

  def _adjacent_traffic(self, row):
    """Traffic in the lane a bypass would use -> (side, count, nearest_gap_m, busy). Looks 60 m back, because
    the cars that matter for a lane change are the ones coming from behind (the recorded row only sees 25 m)."""
    stype, direction = self._bypass_direction()
    if stype is None:
      return None, 0, 999.0, False
    side = 'left' if direction == 'right' else 'right'
    cars = [(g, c) for g, c, _ in self._target_lane_cars(direction) if -60.0 < g < 30.0]
    near = min((g for g, _ in cars), key=abs, default=999.0)
    busy = len(cars) >= 2 or any(-30.0 < g < 15.0 for g, _ in cars)
    return side, len(cars), near, busy

  def _target_lane_cars(self, direction):
    """Cars in the lane we would move into: list of (lon_gap_m, closing_ms, actor). lon_gap > 0 = ahead,
    < 0 = behind (bumper to bumper), 0 = alongside. closing > 0 = a car behind is faster than us."""
    tf = self._vehicle.get_transform()
    fwd = tf.get_forward_vector()
    v = self._vehicle.get_velocity()
    ego_v = v.x * fwd.x + v.y * fwd.y
    ex = self._vehicle.bounding_box.extent.x
    want_left = direction == 'right'           # obstacle on the right -> we go left (lat < 0)
    out = []
    for o in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
      if o.id == self._vehicle.id:
        continue
      ol = o.get_location()
      if ol.distance(tf.location) > 70.0:
        continue
      lon, lat = self._ego_frame(tf, ol)
      if not (1.5 < abs(lat) < 5.5) or (lat < 0) != want_left:
        continue
      of = o.get_transform().get_forward_vector()
      if of.x * fwd.x + of.y * fwd.y < 0.3:
        continue                                 # opposite direction / crossing
      ov = o.get_velocity()
      o_v = ov.x * fwd.x + ov.y * fwd.y
      reach = ex + o.bounding_box.extent.x
      gap = lon - reach if lon > reach else (lon + reach if lon < -reach else 0.0)
      out.append((gap, (o_v - ego_v) if gap <= 0 else (ego_v - o_v), o))
    return out

  def _merge_clear(self, direction, waited_s=0.0, committed=False):
    """Is it OK to (start to / keep) change into the target lane?
    car ahead : 3 m + headway x how much faster we are (a faster car ahead pulls away - no problem)
    car behind: it must be able to slow down comfortably (<= 3 m/s^2) AND keep a time gap:
                3 m + max(headway x closing, closing^2 / 6).  40 km/h vs standing still: ~19 m.
                A slow / stopped car behind is treated as closing at >= 3 m/s (it may pull away).
    Patience: after 10 s of waiting the headway drops from 1.5 s to 1.0 s (the comfortable-braking floor stays).
    committed=True (lane change already under way): only abort for a real danger, otherwise we would creep
    forward and stop again every time the check flickers."""
    hw = self._merge_headway if waited_s < 10.0 else min(self._merge_headway, 1.0)
    worst = None
    for gap, closing, o in self._target_lane_cars(direction):
      if gap == 0.0:
        if committed:
          continue                     # we are already side by side because we are moving over
        return False, 'car alongside'
      if gap > 0:
        need = 1.0 if committed else 3.0 + hw * max(0.0, closing)
        if gap < need:
          return False, f'car {gap:.0f} m ahead'
      else:
        c = max(0.0, closing)
        if committed:
          need = 1.0 + c * c / (2 * 6.0)          # abort only if the car behind would need an emergency stop
        else:
          c = max(c, 3.0)                         # a slow / stopped car behind can pull away any moment
          need = 3.0 + max(hw * c, c * c / (2 * 3.0))
        if -gap < need:
          return False, f'car {-gap:.0f} m behind (needs {need:.0f} m)'
      worst = gap if worst is None or abs(gap) < abs(worst) else worst
    return True, 'clear' if worst is None else f'closest {worst:+.0f} m'

  def _debris_metrics_tick(self):
    """Per-tick bookkeeping during a debris episode: stop time, lane change, closest car in the other lane."""
    st = self._debris_state
    if st is None:
      return
    m = st.setdefault('m', {'stop_s': 0.0, 'lane_changed': False, 'min_gap_m': 999.0})
    v = self._vehicle.get_velocity()
    if (v.x * v.x + v.y * v.y) ** 0.5 < 0.5:
      m['stop_s'] += 0.05
    cmap = CarlaDataProvider.get_map()
    loc = self._vehicle.get_location()
    ewp = cmap.get_waypoint(loc)
    if ewp is None:
      return
    if ewp.lane_id != st['lane_id']:
      m['lane_changed'] = True
    # while we are leaving / out of our lane: closest car in the target lane and how hard cars behind brake
    off_lane = ewp.lane_id != st['lane_id'] or ewp.transform.location.distance(loc) > 0.8
    if 'direction' not in st:
      return
    hist = st.setdefault('spd', {})
    for gap, closing, o in self._target_lane_cars(st['direction']):
      if off_lane and abs(gap) < abs(m['min_gap_m']):
        m['min_gap_m'] = gap
      if gap <= 0 and gap > -30.0:               # car behind us in the target lane
        ov = o.get_velocity()
        tick = self._rec_tick
        h = hist.setdefault(o.id, [])
        if h and tick - h[-1][0] != 1:           # only consecutive ticks (car left the window -> restart)
          h.clear()
        h.append((tick, (ov.x ** 2 + ov.y ** 2) ** 0.5))
        if len(h) > 10:
          h.pop(0)
          dec = (h[0][1] - h[-1][1]) / ((h[-1][0] - h[0][0]) * 0.05)   # mean deceleration over 0.45 s
          if dec > m.get('max_forced_brake', 0.0):
            m['max_forced_brake'] = dec

  def _start_bypass(self, a, dinfo, control, t):
    st = self._debris_state
    st['stype'], st['direction'] = dinfo
    if self._safe_merge and dinfo[0] == 'ParkedObstacle':
      ok, why = self._merge_clear(dinfo[1])
      if not ok:
        st['phase'], st['t_merge'], st['clear_ticks'] = 'merge_wait', t, 0
        self._log('merge_waiting', why=why)
        return self._debris_brake(control)
    st.setdefault('merge_wait_s', 0.0)
    self._hand_debris(a, *dinfo)
    return None

  def _hand_debris(self, a, stype, direction):
    CarlaDataProvider.active_scenarios.append((stype, [a, None, direction, False, 1e9, 1e9, False]))
    self._handled.add(a.id)
    self._log('handed_to_planner', actor=a.type_id, actor_id=a.id, scenario_type=stype, direction=direction)
    self._debris_state['phase'] = 'bypass'

  def _lane_ahead_contains(self, ewp, awp, dist):
    """Does following the ego lane forward (all branches) reach the lane awp is on within dist metres?"""
    key = (awp.road_id, awp.section_id, awp.lane_id)
    if (ewp.road_id, ewp.section_id, ewp.lane_id) == key:
      return True
    frontier, d = [ewp], 0.0
    while d < dist and frontier:
      nxt = []
      for w in frontier:
        nxt.extend(w.next(2.0))
      frontier = nxt[:8]
      d += 2.0
      if any((w.road_id, w.section_id, w.lane_id) == key for w in frontier):
        return True
    return False

  # ------------------------------------------------------------------ construction corridor
  def _ego_lane_refs(self, dist=70.0):
    w = CarlaDataProvider.get_map().get_waypoint(self._vehicle.get_location())
    refs, d = [w], 0.0
    while d < dist:
      n = w.next(1.0)
      if not n:
        break
      w = n[0]
      refs.append(w)
      d += 1.0
    return refs

  def _scan_cones(self):
    """Cones ahead that block our lane -> corridor geometry, or None.
    For every cone: s = metres along our lane, lat = offset from our lane centre (+ = right). The free corridor
    is the widest lateral gap between cone lines and the road edges (our lane right edge, far edge of the
    opposite lane)."""
    world = CarlaDataProvider.get_world()
    tf = self._vehicle.get_transform()
    near = []
    for c in world.get_actors().filter('static.prop.*'):
      if 'cone' not in c.type_id or c.id in self._corr_done:
        continue
      lon, _ = self._ego_frame(tf, c.get_location())
      if 0.0 < lon < self._debris_range:
        near.append(c)
    if len(near) < 4:
      return None
    refs = self._ego_lane_refs(self._debris_range + 10.0)
    pts = []
    for c in near:
      cl = c.get_location()
      k = min(range(len(refs)), key=lambda i: refs[i].transform.location.distance(cl))
      rt = refs[k].transform
      r = rt.get_right_vector()
      lat = (cl.x - rt.location.x) * r.x + (cl.y - rt.location.y) * r.y
      if abs(lat) < 8.0:
        pts.append((float(k), lat, c))
    half_w = self._vehicle.bounding_box.extent.y
    blocking = [p for p in pts if abs(p[1]) < half_w + 0.5]
    if len(blocking) < 2:
      return None                       # cones beside the road, our lane is free
    s0 = min(p[0] for p in blocking)
    zone = [p for p in pts if p[0] >= s0 - 3.0]
    s1 = max(p[0] for p in zone)
    lw = refs[0].lane_width
    left = refs[0].get_left_lane()
    has_opp = left is not None and left.lane_type == carla.LaneType.Driving
    road_left = -lw / 2.0 - (left.lane_width if has_opp else 0.0)
    bounds = sorted([road_left, lw / 2.0] + [p[1] for p in zone])
    lo, hi = max(zip(bounds[:-1], bounds[1:]), key=lambda b: b[1] - b[0])
    width = (hi - lo) - 0.4               # cone radius on both sides
    first = min(zone, key=lambda p: p[0])[2]
    last = max(zone, key=lambda p: p[0])[2]
    lon_first, _ = self._ego_frame(tf, first.get_location())
    return dict(first=first, last=last, cones=[p[2] for p in zone], n=len(zone), s0=s0, length=s1 - s0,
                width=width, center=(lo + hi) / 2.0, centre_line=-lw / 2.0, lane_w=lw,
                gap=lon_first - self._vehicle.bounding_box.extent.x,
                on_centre_line=lo < -lw / 2.0 < hi, uses_opposite=lo < -lw / 2.0)

  def _oncoming(self, lane_w):
    """Opposite-direction vehicles ahead on our road -> (count within 80 m, nearest distance)"""
    tf = self._vehicle.get_transform()
    fwd = tf.get_forward_vector()
    n, near = 0, 999.0
    for o in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
      if o.id == self._vehicle.id:
        continue
      lon, lat = self._ego_frame(tf, o.get_location())
      if not (0.0 < lon < 80.0) or not (-2.0 * lane_w < lat < lane_w / 2.0):
        continue
      of = o.get_transform().get_forward_vector()
      if of.x * fwd.x + of.y * fwd.y > -0.5:
        continue
      n += 1
      near = min(near, lon)
    return n, near

  def _approach_brake(self, control, gap, stop_at=5.0):
    """Approach smoothly (2.5 m/s^2 profile) and stop stop_at metres before a point."""
    if gap < stop_at:
      return carla.VehicleControl(throttle=0.0, brake=1.0, steer=control.steer)
    allowed = (2.0 * 2.5 * (gap - stop_at)) ** 0.5
    v = self._vehicle.get_velocity()
    speed = (v.x ** 2 + v.y ** 2) ** 0.5
    if speed > allowed + 0.5:
      return carla.VehicleControl(throttle=0.0, brake=min(1.0, 0.3 + 0.2 * (speed - allowed)), steer=control.steer)
    return control

  def _shift_route(self, first_loc, last_loc, offset, ramp_m=12.0, pre_m=4.0, post_m=4.0):
    """Move PDM-Lite's route sideways by `offset` m (+ right / - left) from shortly before the first cone to
    shortly after the last one, with smooth ramps. Same mechanism PDM-Lite uses for the official InvadingTurn
    scenario, but in both directions."""
    wp = self._waypoint_planner
    ppm = wp.points_per_meter
    i0 = wp.get_closest_route_index(wp.route_index, first_loc)
    i1 = wp.get_closest_route_index(i0, last_loc)
    ramp, pre, post = int(ramp_m * ppm), int(pre_m * ppm), int(post_m * ppm)
    a = max(wp.route_index, i0 - pre - ramp)
    b = min(len(wp.route_points) - 2, i1 + post + ramp)
    orig = wp.original_route_points
    for idx in range(a, b):
      if idx < i0 - pre:
        f = (idx - (i0 - pre - ramp)) / ramp
      elif idx > i1 + post:
        f = 1.0 - (idx - (i1 + post)) / ramp
      else:
        f = 1.0
      f = min(1.0, max(0.0, f))
      f = f * f * (3.0 - 2.0 * f)
      d = orig[idx + 1, :2] - orig[idx, :2]
      nrm = float(np.linalg.norm(d))
      if nrm < 1e-6:
        continue
      right = np.array([-d[1], d[0]]) / nrm
      wp.route_points[idx, :2] = orig[idx, :2] + right * offset * f
    return i0, i1

  def _corridor_summary(self, g, onc_n, onc_d):
    return (f"construction zone ahead: {g['n']} traffic cones over {g['length']:.0f} m, starting {g['gap']:.0f} m ahead. "
            f"Cones stand inside our lane, so our lane is blocked.\n"
            f"free corridor between the cone lines: {g['width']:.1f} m wide, its centre is "
            f"{abs(g['center']):.1f} m to the {'left' if g['center'] < 0 else 'right'} of our lane centre"
            f"{' (on the centre line, partly in the opposite lane)' if g['on_centre_line'] else ''}; "
            f"our car is {2 * self._vehicle.bounding_box.extent.y:.1f} m wide. This corridor is not in the map.\n"
            f"oncoming vehicles within 80 m: {onc_n}" + (f" (nearest {onc_d:.0f} m)" if onc_n else "") + "\n")

  def _ask_corridor(self, row, t):
    st = self._corr
    g = self._scan_cones() or st['g']
    st['g'] = g
    onc_n, onc_d = self._oncoming(g['lane_w'])
    summary = self._corridor_summary(g, onc_n, onc_d) + build_summary(row)
    drow = dict(row)
    drow.update(kind='corridor', corr_width=g['width'], corr_center=g['center'], ego_width=2 * self._vehicle.bounding_box.extent.y,
                oncoming_n=onc_n, oncoming_dist=onc_d)
    self._calls += 1
    self._log('trigger', call=self._calls, by='cones_detected', kind='corridor', n_cones=g['n'], gap_m=round(g['gap'], 1),
              corridor_width_m=round(g['width'], 2), corridor_center_m=round(g['center'], 2), oncoming=onc_n, summary=summary)
    st['t'] = t
    self._advisor.request(self._jpeg(), summary, drow)

  def _corridor_step(self, control, row, t):
    st = self._corr
    if st is None:
      if t < 2.0 or self._pending_actor is not None or self._debris_state is not None or self._backing_up is not None:
        return None
      g = self._scan_cones()
      if g is None:
        return None
      self._corr = st = {'g': g, 'phase': 'asking', 't_seen': t, 'stop_s': 0.0, 'min_clear': 99.0}
      self._log('cones_seen', n=g['n'], gap_m=round(g['gap'], 1), width_m=round(g['width'], 2), center_m=round(g['center'], 2))
      self._ask_corridor(row, t)
      return self._approach_brake(control, g['gap'])

    g = st['g']
    if not g['first'].is_alive or not g['last'].is_alive:
      self._corr = None
      return None
    tf = self._vehicle.get_transform()
    v = self._vehicle.get_velocity()
    speed = (v.x ** 2 + v.y ** 2) ** 0.5
    if speed < 0.5:
      st['stop_s'] += 0.05
    lon_first, _ = self._ego_frame(tf, g['first'].get_location())
    lon_last, _ = self._ego_frame(tf, g['last'].get_location())
    gap = lon_first - self._vehicle.bounding_box.extent.x

    if st['phase'] == 'asking':
      res = self._advisor.poll()
      if res is None:
        return self._approach_brake(control, gap)
      self._log('llm_answer', kind='corridor', sim_wait_s=round(t - st['t'], 2), gap_m=round(gap, 1), **res)
      if res['decision'] in ('BYPASS', 'PROCEED'):
        i0, i1 = self._shift_route(g['first'].get_location(), g['last'].get_location(), g['center'])
        self._log('route_shifted', offset_m=round(g['center'], 2), route_idx=[int(i0), int(i1)])
        st['phase'] = 'go'
        return None
      st['phase'], st['t_wait'] = 'wait', t
      return self._approach_brake(control, gap)
    if st['phase'] == 'wait':
      if t - st['t_wait'] >= 3.0 and self._calls < self._max_calls:
        st['phase'] = 'asking'
        self._ask_corridor(row, t)
      return self._approach_brake(control, gap)
    if st['phase'] == 'go':
      # closest cone beside the car while passing (lateral clearance, car side to cone centre - 0.2 m)
      ex, ey = self._vehicle.bounding_box.extent.x, self._vehicle.bounding_box.extent.y
      for c in g['cones']:
        lon, lat = self._ego_frame(tf, c.get_location())
        if abs(lon) < ex + 0.3:
          st['min_clear'] = min(st['min_clear'], abs(lat) - ey - 0.2)
      if lon_last < -8.0:
        self._log('corridor_passed', took_s=round(t - st['t_seen'], 1), stop_s=round(st['stop_s'], 1),
                  min_cone_clearance_m=round(st['min_clear'], 2))
        for c in g['cones']:
          self._corr_done.add(c.id)
        self._corr = None
        return None
      if lon_first < 15.0 and speed > 20.0 / 3.6:       # slow through the zone (<= 20 km/h)
        return carla.VehicleControl(throttle=0.0, brake=0.3, steer=control.steer)
      return None
    return None

  def _debris_gap(self, a):
    tf = self._vehicle.get_transform()
    lon, _ = self._ego_frame(tf, a.get_location())
    return lon - a.bounding_box.extent.x - self._vehicle.bounding_box.extent.x

  def _debris_brake(self, control):
    """Safe default while no bypass/proceed is under way: approach smoothly and stop ~5 m before the object.
    Speed limit follows a comfortable 2.5 m/s^2 braking profile, so far away PDM-Lite drives normally."""
    st = self._debris_state
    if st is None or not st['actor'].is_alive:
      return control
    gap = self._debris_gap(st['actor'])
    stop_at = 5.0
    if gap < stop_at:
      return carla.VehicleControl(throttle=0.0, brake=1.0, steer=control.steer)
    allowed = (2.0 * 2.5 * (gap - stop_at)) ** 0.5
    v = self._vehicle.get_velocity()
    speed = (v.x ** 2 + v.y ** 2) ** 0.5
    if speed > allowed + 0.5:
      return carla.VehicleControl(throttle=0.0, brake=min(1.0, 0.3 + 0.2 * (speed - allowed)), steer=control.steer)
    return control

  def _ask_debris(self, a, row, t):
    e = a.bounding_box.extent
    gap = self._debris_gap(a)
    L, W, H = 2 * e.x, 2 * e.y, 2 * e.z
    name = a.type_id.replace('static.prop.', '')
    if self._perception == 'noisy':
      n = lambda sd: max(0.3, 1.0 + self._prng.gauss(0.0, sd))
      L, W, H = L * n(0.3), W * n(0.3), H * n(0.3)       # size estimated from the image: +-30 %
      gap = gap * n(0.1)                                 # distance estimate: +-10 %
      name = 'unknown object'                            # the detector does not know what it is
    jpeg = self._jpeg()
    img_name = f'call{self._calls}_debris_t{t:.1f}.jpg'
    if jpeg:
      with open(os.path.join(self._ev_dir, img_name), 'wb') as f:
        f.write(jpeg)
    approx = 'about ' if self._perception == 'noisy' else ''
    summary = (f"object lying on the road in the ego lane: {name} "
               f"(a static object, not a vehicle), {approx}{L:.2f} x {W:.2f} m, height {approx}{H:.2f} m, "
               f"{approx}{gap:.1f} m ahead\n")
    side, n, near, busy = self._adjacent_traffic(row)
    if side is None:
      summary += "no adjacent lane to go around\n"
    else:
      summary += (f"{side} lane (the one a bypass would use): {n} vehicles nearby, closest {near:+.0f} m "
                  f"-> {'BUSY, changing lanes is risky' if busy else 'free'}\n")
    summary += build_summary(row)
    drow = dict(row)
    drow.update(kind='debris', obj_type=name, obj_h=H, obj_w=W, obj_l=L, adj_busy=busy, adj_count=n,
                adj_possible=side is not None)
    self._log('trigger', call=self._calls, by='debris_detected', kind='debris', perception=self._perception,
              front=a.type_id, front_id=a.id, gap_m=round(gap, 1), obj_h_true=round(2 * e.z, 2), obj_h_est=round(H, 2),
              image=img_name if jpeg else None, summary=summary)
    self._debris_state['t'] = t
    self._advisor.request(jpeg, summary, drow)

  def _debris_step(self, control, row, t):
    """Returns a control to use, or None to let the normal (vehicle) logic run."""
    st = self._debris_state
    if st is None:
      if self._pending_actor is not None or self._snap_wait is not None or self._backing_up is not None:
        return None
      if t < 2.0:                          # perception warm-up: scenario traffic is spawned in the first second
        return None
      hit = self._scan_debris()
      if hit is None:
        return None
      gap, lat, a = hit
      ewp = CarlaDataProvider.get_map().get_waypoint(self._vehicle.get_location())
      self._debris_state = st = {'actor': a, 'phase': 'asking', 't': t, 't_seen': t, 'lane_id': ewp.lane_id}
      side, n, near, busy = self._adjacent_traffic(row)
      self._log('debris_seen', actor=a.type_id, gap_m=round(gap, 1), lat_m=round(lat, 2), policy=self._debris_policy,
                adj_side=side, adj_count=n, adj_nearest_m=round(near, 1), adj_busy=busy)
      if self._debris_policy == 'always_bypass':
        self._log('trigger', call=0, by='always_bypass', kind='debris', front=a.type_id, front_id=a.id, gap_m=round(gap, 1))
        dinfo = self._bypass_direction()
        if dinfo[0] is None:
          self._log('bypass_impossible', reason='no adjacent driving lane')
          st['phase'] = 'stuck'
          return self._debris_brake(control)
        return self._start_bypass(a, dinfo, control, t)
      self._calls += 1
      if self._use_cam:
        try:
          self._start_snapshot()
          self._snap_wait = (a, t, 'debris')
          return self._debris_brake(control)
        except Exception as ex:
          print(f'>> [Intervention] camera spawn failed ({ex}) -> text only', flush=True)
      self._ask_debris(a, row, t)
      return self._debris_brake(control)

    a = st['actor']
    if not a.is_alive:
      self._debris_state = None
      return None
    self._debris_metrics_tick()
    gap = self._debris_gap(a)
    if st['phase'] == 'stuck':
      return self._debris_brake(control)
    if st['phase'] == 'merge_wait':
      ok, why = self._merge_clear(st['direction'], st.get('merge_wait_s', 0.0) + (t - st['t_merge']))
      st['clear_ticks'] = st.get('clear_ticks', 0) + 1 if ok else 0
      if st['clear_ticks'] >= 3:
        self._log('merge_gap_found', waited_s=round(t - st['t_merge'], 1), why=why)
        st['merge_wait_s'] = st.get('merge_wait_s', 0.0) + (t - st['t_merge'])
        st['committed'] = True
        self._hand_debris(a, st['stype'], st['direction'])
        return None
      if t - st.get('t_why', -1e9) > 2.0:
        st['t_why'] = t
        self._log('merge_waiting', why=why)
      return self._debris_brake(control)
    if st['phase'] == 'bypass' and self._safe_merge and st.get('stype') == 'ParkedObstacle' and not st.get('merged'):
      # PDM-Lite starts the lane change on its own later -> keep checking the target lane until we are in it
      ewp = CarlaDataProvider.get_map().get_waypoint(self._vehicle.get_location())
      if ewp.lane_id != st['lane_id']:
        st['merged'] = True
        self._log('merged', gap_to_object_m=round(gap, 1), merge_wait_s=round(st.get('merge_wait_s', 0.0), 1))
      elif gap < 25.0:
        committed = st.get('committed', False)
        ok, why = self._merge_clear(st['direction'], st.get('merge_wait_s', 0.0))
        # once we have decided to go we do NOT abort: stopping half-way in the target lane is what forced the
        # car behind into hard braking (it had already started to react to us)
        if not committed:
          st['clear_ticks'] = st.get('clear_ticks', 0) + 1 if ok else 0
          if st['clear_ticks'] >= 3:
            st['committed'] = True
            self._log('merge_go', why=why, gap_to_object_m=round(gap, 1), waited_s=round(st.get('merge_wait_s', 0.0), 1))
        if not st.get('committed', False):
          st['merge_wait_s'] = st.get('merge_wait_s', 0.0) + 0.05
          if t - st.get('t_why', -1e9) > 2.0:
            st['t_why'] = t
            self._log('merge_hold', why=why, gap_to_object_m=round(gap, 1))
          return self._debris_brake(control)    # approach smoothly and wait ~5 m before the object
    if st['phase'] in ('proceed', 'bypass'):
      if gap < -8.0:                       # object is behind us -> done
        m = st.get('m', {})
        self._log('debris_passed', actor=a.type_id, decision=st['phase'], policy=self._debris_policy,
                  took_s=round(t - st['t_seen'], 1), stop_s=round(m.get('stop_s', 0.0), 1),
                  lane_changed=m.get('lane_changed', False), min_gap_other_lane_m=round(m.get('min_gap_m', 999.0), 1),
                  merge_wait_s=round(st.get('merge_wait_s', 0.0), 1),
                  max_forced_brake_ms2=round(m.get('max_forced_brake', 0.0), 1))
        self._debris_done.add(a.id)
        self._debris_state = None
      return None
    if st['phase'] == 'asking':
      res = self._advisor.poll()
      if res is None:
        return self._debris_brake(control)
      self._log('llm_answer', kind='debris', sim_wait_s=round(t - st['t'], 2), gap_m=round(gap, 1), **res)
      dec = res['decision']
      if dec == 'PROCEED':
        st['phase'] = 'proceed'
        return None
      if dec == 'BYPASS':
        dinfo = self._bypass_direction()
        if dinfo[0] is None:
          self._log('bypass_impossible', reason='no adjacent driving lane')
          st['phase'] = 'wait'
          st['t_wait'] = t
          return self._debris_brake(control)
        return self._start_bypass(a, dinfo, control, t)
      st['phase'] = 'wait'                 # WAIT
      st['t_wait'] = t
      return self._debris_brake(control)
    if st['phase'] == 'wait':
      if t - st['t_wait'] >= self._reask and self._calls < self._max_calls:
        self._calls += 1
        st['phase'] = 'asking'
        self._ask_debris(a, row, t)
      return self._debris_brake(control)
    return None

  def destroy(self, results=None):
    self._stop_snapshot()
    try:
      self._log('end', calls=self._calls, handed=len(self._handled))
      self._ev_file.close()
    except Exception:
      pass
    super().destroy(results)
