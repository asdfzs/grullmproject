"""
ClearEgoLaneAhead (GRU/LLM deadlock project).

Background traffic sometimes drives in front of the ego and then stops behind our scenario car, so the
ego ends up waiting behind an ordinary background car (wrong data). The leaderboard's own
LeaveSpaceInFront does not always fire (e.g. near junctions). This behavior removes NON-scenario
vehicles that are in the ego lane BETWEEN the ego and the scenario's first car, while active.
It never returns SUCCESS, so put it in a SUCCESS_ON_ONE parallel next to the real behavior.

v2 (bug fix): it is active ONLY during the approach - it switches itself off for good once the ego is within
`stop_dist` (20 m) of the scenario car, and it only removes cars that are in the SAME map lane as the scenario
car (or the ego's own lane) - never a neighbouring lane. v1 kept running during the bypass: when the ego swung into the other lane, oncoming/adjacent cars
counted as "in front of the ego" and were deleted just before a near-collision.
"""
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.timer import GameTime


class ClearEgoLaneAhead(AtomicBehavior):

    def __init__(self, ego, target, keep_ids, active_time=60.0, lane_half_width=2.0, stop_dist=20.0,
                 name="ClearEgoLaneAhead"):
        super().__init__(name)
        self._ego = ego
        self._target = target
        self._keep = set(keep_ids)
        self._active_time = active_time
        self._half = lane_half_width
        self._t0 = None
        self._stop_dist = stop_dist
        self._done = False
        self._lane = None

    def initialise(self):
        self._t0 = GameTime.get_time()
        super().initialise()

    def update(self):
        if self._t0 is None:
            self._t0 = GameTime.get_time()
        if self._done or GameTime.get_time() - self._t0 > self._active_time:
            return py_trees.common.Status.RUNNING
        try:
            if not self._target.is_alive:
                return py_trees.common.Status.RUNNING
            cmap = CarlaDataProvider.get_map()
            if self._lane is None:
                twp = cmap.get_waypoint(self._target.get_location())
                self._lane = (twp.road_id, twp.lane_id)
            if self._ego.get_location().distance(self._target.get_location()) < self._stop_dist:
                self._done = True          # approach finished -> never touch traffic again (bypass is real driving)
                return py_trees.common.Status.RUNNING
            tf = self._ego.get_transform()
            f, r = tf.get_forward_vector(), tf.get_right_vector()

            def frame(loc):
                dx, dy = loc.x - tf.location.x, loc.y - tf.location.y
                return dx * f.x + dy * f.y, dx * r.x + dy * r.y

            tgt_lon, _ = frame(self._target.get_location())
            ewp = cmap.get_waypoint(self._ego.get_location())
            my_lanes = {self._lane, (ewp.road_id, ewp.lane_id)}
            for a in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
                if a.id == self._ego.id or a.id in self._keep:
                    continue
                lon, lat = frame(a.get_location())
                if 0.0 < lon < tgt_lon - 1.0 and abs(lat) < self._half:
                    awp = cmap.get_waypoint(a.get_location())
                    if awp is None or (awp.road_id, awp.lane_id) not in my_lanes:
                        continue           # only the ego's own lane / the scenario car's lane - never a side lane
                    print(f'>> [ClearEgoLaneAhead] removed background {a.type_id} {lon:.1f} m ahead of ego', flush=True)
                    CarlaDataProvider.remove_actor_by_id(a.id)
                    if a.is_alive:
                        a.destroy()
        except Exception as e:
            print(f'>> [ClearEgoLaneAhead] {e}', flush=True)
        return py_trees.common.Status.RUNNING
