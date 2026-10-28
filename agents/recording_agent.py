"""
RecordingAgent = PDM-Lite (DataAgent) + per-tick feature logging for the GRU deadlock project.

Driving is 100% PDM-Lite (unchanged). Every tick (20 Hz) we additionally write one CSV row of
"object-list level" features (the kind a real perception stack outputs), computed from CARLA.

Environment variables (set by run_record.ps1):
  RECORD_TAG     e.g. "165_stalled"            -> used in the CSV file name
  RECORD_CLASS   "P" (needs LLM) or "N" (resolves by itself)
  RECORD_DIR     output folder (default: <WORK_DIR>/results/features)
  MAX_STOP_SEC   end the episode after the ego has been stopped this long (default 40 s).
                 Deadlock episodes otherwise run until AgentBlocked (180 s) - wasteful for data.
"""
import os
import csv
import math
import datetime

import carla

from data_agent import DataAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
  return 'RecordingAgent'


COLUMNS = [
    'tick', 'sim_time_s', 'tag', 'episode_class',
    'ego_speed_kmh', 'lon_accel_ms2', 'yaw_rate_degs', 'stop_duration_s',
    'front_dist_m', 'front_rel_speed_ms', 'front_speed_ms', 'front_width_m', 'front_class',
    'front_hazard_on', 'front_stopped_duration_s', 'front2_dist_m', 'front2_speed_ms',
    'left_occupied', 'left_nearest_rel_s_m', 'left_rel_speed_ms', 'left_count', 'left_mean_speed_ms',
    'right_occupied', 'right_nearest_rel_s_m', 'right_rel_speed_ms', 'right_count', 'right_mean_speed_ms',
    'traffic_light_red', 'traffic_light_dist_m', 'at_junction',
]

SAME_LANE_LAT = 1.8          # |lateral| below this = same lane as ego (m)
ADJ_LANE_LAT = (1.8, 5.5)    # adjacent lane band (m)
FRONT_RANGE = 50.0           # how far ahead we look for the front object (m)
ADJ_RANGE = (-25.0, 35.0)    # longitudinal window for adjacent lanes (m)
NONE_DIST = 999.0
STOP_KMH = 3.0   # PDM-Lite makes tiny ~1.4 km/h nudges every ~9 s while waiting; ignore them
DT = 0.05


class RecordingAgent(DataAgent):

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager)
    self._rec_tag = os.environ.get('RECORD_TAG', 'untagged')
    self._rec_class = os.environ.get('RECORD_CLASS', '?')
    self._max_stop = float(os.environ.get('MAX_STOP_SEC', '40'))
    out_dir = os.environ.get('RECORD_DIR', os.path.join(os.environ.get('WORK_DIR', '.'), 'results', 'features'))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self._rec_path = os.path.join(out_dir, f'{self._rec_tag}_{stamp}.csv')
    self._rec_file = open(self._rec_path, 'w', newline='', encoding='utf-8')
    self._rec_writer = csv.writer(self._rec_file)
    self._rec_writer.writerow(COLUMNS)
    self._rec_tick = 0
    self._prev_speed = None
    self._prev_yaw = None
    self._stop_dur = 0.0
    self._front_stop = {}   # actor id -> seconds stopped
    self._last_row = None
    # END_AFTER_RELEASE=1 (queue routes): end the episode once the queue has released and the ego is
    # driving again, so a later, unrelated stop (traffic at the next junction) doesn't pollute the data.
    self._end_after_release = os.environ.get('END_AFTER_RELEASE', '0') == '1'
    self._had_stop = False
    self._moving_after = 0.0
    self._last_front_actor = None
    print(f'>> [RecordingAgent] logging features to {self._rec_path} (class={self._rec_class})', flush=True)

  # ------------------------------------------------------------------ helpers
  @staticmethod
  def _speed(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

  def _ego_frame(self, ego_tf, loc):
    """(longitudinal, lateral) of loc in the ego frame. lateral > 0 = right (CARLA convention)."""
    f = ego_tf.get_forward_vector()
    r = ego_tf.get_right_vector()
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    return dx * f.x + dy * f.y, dx * r.x + dy * r.y

  def _red_light_ahead(self, world, ego):
    """(red_or_yellow_flag, distance_m) of the next traffic light ON THE EGO ROUTE.
    Uses PDM-Lite's route planner (it pre-computes the next light for every route point), so it also
    works when the ego waits in a queue far behind the stop line (the earlier waypoint-based check
    missed those cases)."""
    try:
      wp_planner = self._waypoint_planner
      idx = min(wp_planner.route_index, len(wp_planner.next_traffic_lights) - 1)
      tl = wp_planner.next_traffic_lights[idx]
      dist = float(wp_planner.distances_to_next_traffic_lights[idx])
      if tl is not None and dist < 80.0:
        red = 1 if tl.get_state() in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow) else 0
        return red, dist
    except Exception:
      pass
    return 0, NONE_DIST

  # ------------------------------------------------------------------ main
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)
    try:
      self._record()
    except Exception as e:  # never break driving because of logging
      print(f'>> [RecordingAgent] logging error: {e}', flush=True)
    return control

  def _record(self):
    ego = self._vehicle
    world = CarlaDataProvider.get_world()
    ego_tf = ego.get_transform()
    ego_fwd = ego_tf.get_forward_vector()
    speed = self._speed(ego)
    yaw = ego_tf.rotation.yaw

    lon_acc = 0.0 if self._prev_speed is None else (speed - self._prev_speed) / DT
    yaw_rate = 0.0 if self._prev_yaw is None else ((yaw - self._prev_yaw + 180) % 360 - 180) / DT
    self._prev_speed, self._prev_yaw = speed, yaw
    self._stop_dur = self._stop_dur + DT if speed * 3.6 < STOP_KMH else 0.0

    front = []                      # (lon, actor, speed)
    left, right = [], []            # (lon, rel_speed, speed)
    for a in world.get_actors():
      if a.id == ego.id or not (a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.')):
        continue
      loc = a.get_location()
      if loc.distance(ego_tf.location) > 60.0:
        continue
      lon, lat = self._ego_frame(ego_tf, loc)
      a_speed = self._speed(a)
      if 0.0 < lon < FRONT_RANGE and abs(lat) < SAME_LANE_LAT:
        front.append((lon, a, a_speed))
        continue
      if a.type_id.startswith('vehicle.') and ADJ_RANGE[0] < lon < ADJ_RANGE[1] \
          and ADJ_LANE_LAT[0] <= abs(lat) <= ADJ_LANE_LAT[1]:
        f2 = a.get_transform().get_forward_vector()
        if ego_fwd.x * f2.x + ego_fwd.y * f2.y < 0.3:
          continue  # opposite / crossing direction
        (right if lat > 0 else left).append((lon, a_speed - speed, a_speed))
    front.sort(key=lambda e: e[0])

    # update "how long has this front actor been stopped"
    seen = set()
    for _, a, s in front:
      seen.add(a.id)
      self._front_stop[a.id] = self._front_stop.get(a.id, 0.0) + DT if s < 0.3 else 0.0
    for k in list(self._front_stop):
      if k not in seen:
        del self._front_stop[k]

    if front:
      lon, a, s = front[0]
      ext = a.bounding_box.extent
      front_dist = max(0.0, lon - ext.x - ego.bounding_box.extent.x)
      front_rel = s - speed
      front_width = 2 * ext.y
      if a.type_id.startswith('walker.'):
        fclass, hazard = 'pedestrian', 0
      else:
        fclass = 'vehicle_stopped' if s < 0.3 else 'vehicle_moving'
        try:
          ls = a.get_light_state()
          hz = carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker
          hazard = 1 if (int(ls) & int(hz)) == int(hz) else 0
        except Exception:
          hazard = 0
      fstop = self._front_stop.get(a.id, 0.0)
    else:
      front_dist, front_rel, s, front_width, fclass, hazard, fstop = NONE_DIST, 0.0, 0.0, 0.0, 'none', 0, 0.0
    if len(front) > 1:
      front2_dist, front2_speed = front[1][0], front[1][2]
    else:
      front2_dist, front2_speed = NONE_DIST, 0.0

    def summarize(entries):
      if not entries:
        return 0, NONE_DIST, 0.0, 0, 0.0
      n = min(entries, key=lambda e: abs(e[0]))
      return 1, n[0], n[1], len(entries), sum(e[2] for e in entries) / len(entries)

    lo, ls_, lr, lc, lm = summarize(left)
    ro, rs, rr, rc, rm = summarize(right)
    red, tl_dist = self._red_light_ahead(world, ego)
    junction = 1 if world.get_map().get_waypoint(ego_tf.location).is_junction else 0

    self._rec_tick += 1
    self._last_front_actor = front[0][1] if front else None   # used by InterventionAgent
    row = [
        self._rec_tick, round(self._rec_tick * DT, 2), self._rec_tag, self._rec_class,
        round(speed * 3.6, 2), round(lon_acc, 3), round(yaw_rate, 3), round(self._stop_dur, 2),
        round(front_dist, 2), round(front_rel, 2), round(s, 2), round(front_width, 2), fclass,
        hazard, round(fstop, 2), round(front2_dist, 2), round(front2_speed, 2),
        lo, round(ls_, 2), round(lr, 2), lc, round(lm, 2),
        ro, round(rs, 2), round(rr, 2), rc, round(rm, 2),
        red, round(tl_dist, 1), junction,
    ]
    self._last_row = dict(zip(COLUMNS, row))                    # used by InterventionAgent
    self._rec_writer.writerow(row)
    if self._rec_tick % 100 == 0:
      self._rec_file.flush()

    if self._end_after_release:
      if self._stop_dur >= 3.0:
        self._had_stop = True
      if self._had_stop and speed * 3.6 > 15.0:
        self._moving_after += DT
      if self._moving_after >= 3.0:
        print('>> [RecordingAgent] queue released and ego driving again -> ending episode', flush=True)
        self._rec_file.flush()
        self._rec_file.close()
        os._exit(0)

    if self._stop_dur >= self._max_stop:
      print(f'>> [RecordingAgent] ego stopped for {self._stop_dur:.1f}s >= MAX_STOP_SEC -> ending episode', flush=True)
      self._rec_file.flush()
      self._rec_file.close()
      os._exit(0)

  def destroy(self, results=None):
    try:
      self._rec_file.flush()
      self._rec_file.close()
    except Exception:
      pass
    super().destroy(results)
