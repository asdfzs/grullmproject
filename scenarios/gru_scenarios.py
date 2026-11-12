#!/usr/bin/env python
"""
Scenarios added for the GRU + LLM deadlock project (data collection).

  StalledCar    (P-2, label "LLM needed"): same geometry as BrokenDownCar / ParkedObstacle
                but NO hazard lights and NO warning triangle. Never moves.
  QueueRelease  (N-2, label "resolves by itself"): 2-3 cars stopped in the ego lane (a queue,
                no hazard lights). Once the ego has stopped behind them, they wait a random
                5-20 s and then drive off (autopilot). Mimics a signal / congestion queue.

Neither scenario registers itself in CarlaDataProvider.active_scenarios, so PDM-Lite
treats the cars as ordinary vehicles (it stops behind them and waits).

Distances are jittered randomly every run so repeated episodes differ.

v2 (shortcut removal): the GRU learned "a 2nd car visible ahead = queue". So now
  StalledCar   : 50% of runs also get a 2nd parked car in front of the stalled one (param lead_car: 0/1/-1=random)
  QueueRelease : 1, 2 or 3 cars (param num_cars, 0 = random) -> a single car that drives off also exists
"""
import os
import random

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, Idle, WaitForever, ScenarioTimeout, ChangeAutoPilot, HandBrakeVehicle, StopVehicle)
try:
    from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorFlow
except ImportError:          # the plain scenario_runner copy has no ActorFlow
    ActorFlow = None
from srunner.scenariomanager.scenarioatomics.atomic_criteria import ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation, WaitUntilInFrontPosition)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeOppositeBehavior
from srunner.scenarios.route_obstacles import get_value_parameter
from srunner.scenarios.broken_down_car import BrokenDownCar
from srunner.tools.lane_clear import ClearEgoLaneAhead

_rng = random.Random()  # seeded from OS entropy -> differs every run


if ActorFlow is not None:
    class SparseFlow(ActorFlow):
        """ActorFlow whose cars keep a random distance (6-35 m) to the car in front. The plain flow bunches into
        an evenly spaced platoon (~3 s apart) that never leaves a usable gap; real traffic has uneven gaps."""
        def _spawn_actor(self, transform):
            n = len(self._actor_list)
            res = super()._spawn_actor(transform)
            if len(self._actor_list) > n:
                try:
                    self._tm.distance_to_leading_vehicle(self._actor_list[-1], _rng.uniform(6.0, 35.0))
                except Exception:
                    pass
            return res
else:
    SparseFlow = None


class StalledCar(BrokenDownCar):
    """P-2: stalled car without any visual cue (no hazard lights, no triangle)."""

    def _initialize_actors(self, config):
        self._distance = self._distance + _rng.uniform(-8.0, 8.0)
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        self._vehicle_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        car = self._spawn_obstacle(self._vehicle_wp, 'vehicle.*')
        car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
        car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(car)

        lead = get_value_parameter(config, 'lead_car', int, -1)
        if lead == 1 or (lead == -1 and _rng.random() < 0.5):
            lead_wp = self._move_waypoint_forward(self._vehicle_wp, _rng.uniform(7.0, 9.0))
            try:
                car2 = self._spawn_obstacle(lead_wp, 'vehicle.*')
                car2.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
                car2.apply_control(carla.VehicleControl(hand_brake=True))
                self.other_actors.append(car2)
                print(f">> [StalledCar] + 2nd parked car {car2.type_id} in front of it")
            except Exception as e:
                print(f">> [StalledCar] could not spawn 2nd car: {e}")

        self._spawn_cues(config)

        self._end_wp = self._move_waypoint_forward(self._vehicle_wp, self._end_distance)
        # NOT registered in CarlaDataProvider.active_scenarios
        print(f">> [StalledCar] {car.type_id} (id={car.id}) {self._distance:.1f}m ahead, "
              f"no hazard lights, NOT registered for the expert")


    def _spawn_cues(self, config):
        """v3: optional visual signs of a breakdown, visible ONLY in the camera photo (not in the text summary).
        person=1   : the driver stands on the shoulder next to the rear of the stalled car, facing it
        triangle=X : a warning triangle on the shoulder X m behind the stalled car's rear (outside our lane). The ego stops
                     ~4 m behind the car and the photo is taken there, so X must be < ~3 m to be in the picture."""
        person = get_value_parameter(config, 'person', int, 0)
        tri_m = get_value_parameter(config, 'triangle', float, 0.0)
        if not person and tri_m <= 0:
            return
        wp = self._vehicle_wp
        base = wp.transform.location
        yaw = wp.transform.rotation.yaw
        fwd = wp.transform.get_forward_vector()
        rgt = wp.transform.get_right_vector()
        sgn = 1.0 if self._direction == 'right' else -1.0
        lw = wp.lane_width
        car_lat = self._offset * lw / 2.0                     # stalled car centre, from the lane centre

        def at(lon, lat, dz):
            return carla.Location(x=base.x + fwd.x * lon + rgt.x * sgn * lat,
                                  y=base.y + fwd.y * lon + rgt.y * sgn * lat, z=base.z + dz)

        if person:
            walker = None
            for lat in (car_lat + 1.3, car_lat + 1.6, car_lat + 2.0, car_lat + 2.5):
                tf = carla.Transform(at(-2.4, lat, 1.0), carla.Rotation(yaw=yaw - 90.0 * sgn))
                try:
                    walker = CarlaDataProvider.request_new_actor('walker.pedestrian.*', tf, rolename='scenario')
                except Exception:
                    walker = None
                if walker:
                    break
            if walker:
                self.other_actors.append(walker)
                print(f">> [StalledCar] + person {walker.type_id} standing next to it ({lat:.1f} m from lane centre)")
            else:
                print(">> [StalledCar] !! could not spawn the person")

        if tri_m > 0:
            tf = carla.Transform(at(-(2.3 + tri_m), lw / 2.0 + 0.3, 0.2), carla.Rotation(yaw=yaw + 90.0))
            try:
                tri = CarlaDataProvider.request_new_actor('static.prop.warningaccident', tf)
            except Exception:
                tri = None
            if tri:
                tri.set_simulate_physics(False)
                self.other_actors.append(tri)
                print(f">> [StalledCar] + warning triangle {tri_m:.0f} m behind it (on the shoulder)")
            else:
                print(">> [StalledCar] !! could not spawn the warning triangle")


class QueueRelease(BasicScenario):
    """N-2: a short queue of stopped cars that drives off after a random wait."""

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._distance = get_value_parameter(config, 'distance', float, 40) + _rng.uniform(-8.0, 8.0)
        n = get_value_parameter(config, 'num_cars', int, 0)
        self._num_cars = n if n > 0 else _rng.choice([2, 3])   # v3: a queue has a reason ahead -> no lone car
        self._gap = _rng.uniform(7.0, 9.0)           # spacing between queued cars (m)
        hmin = get_value_parameter(config, 'hold_min', float, 5.0)
        hmax = get_value_parameter(config, 'hold_max', float, 20.0)
        self._hold = _rng.uniform(hmin, hmax)         # how long the queue waits after ego stops (s)
        self._trigger_distance = 12.0                 # ego this close to the last car = "stopped behind it"
        self._scenario_timeout = 240
        self._queue = []
        super().__init__("QueueRelease", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable)

    def _move_forward(self, wp, distance):
        dist, nxt = 0.0, wp
        while dist < distance:
            n = nxt.next(1.0)
            if not n:
                break
            nxt = n[0]
            dist += 1.0
        return nxt

    def _initialize_actors(self, config):
        start_wp = self._map.get_waypoint(config.trigger_points[0].location)
        # first queued car (the one nearest to the ego) and the ones in front of it
        for i in range(self._num_cars):
            wp = self._move_forward(start_wp, self._distance + i * self._gap)
            tf = wp.transform
            tf.location.z += 0.5
            car = CarlaDataProvider.request_new_actor(
                'vehicle.*', tf, rolename='scenario', attribute_filter={'base_type': 'car', 'generation': 2})
            if car is None:
                continue
            car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
            car.apply_control(carla.VehicleControl(hand_brake=True))
            self._queue.append(car)
            self.other_actors.append(car)
        if not self._queue:
            raise ValueError("QueueRelease: could not spawn any queued car")
        self._last_car = self._queue[0]   # closest to ego
        self._end_wp = self._move_forward(start_wp, self._distance + self._num_cars * self._gap + 40)
        print(f">> [QueueRelease] {len(self._queue)} cars from {self._distance:.1f}m, "
              f"gap {self._gap:.1f}m, release {self._hold:.1f}s after ego stops")

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="QueueRelease")
        if self.route_mode:
            # keep background traffic out of the queue area (same as ParkedObstacle does)
            root.add_child(LeaveSpaceInFront(self._distance + self._num_cars * self._gap + 20))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._last_car.get_location(), self._trigger_distance))
        behavior.add_child(Idle(self._hold))
        release = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        for car in self._queue:
            seq = py_trees.composites.Sequence()
            seq.add_child(HandBrakeVehicle(car, False))
            seq.add_child(ChangeAutoPilot(car, True, parameters={"auto_lane_change": False}))
            release.add_child(seq)
        behavior.add_child(release)
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        if self.route_mode:
            end_condition.add_child(ClearEgoLaneAhead(self.ego_vehicles[0], self._last_car,
                                                      [a.id for a in self.other_actors]))
        root.add_child(end_condition)
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))
        return root

    def _create_test_criteria(self):
        return [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]

    def __del__(self):
        self.remove_all_actors()


class StopAndGo(BasicScenario):
    """N-3 (does NOT need the LLM): congested stop-and-go traffic.
    1-2 cars in the ego lane repeatedly drive a little (autopilot, slow) and stop for 2-8 s (no red light).
    The ego keeps stopping behind a stopped car - the typical source of useless LLM calls."""

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._distance = get_value_parameter(config, 'distance', float, 25) + _rng.uniform(-5.0, 5.0)
        self._num_cars = _rng.choice([1, 2])
        self._cycles = get_value_parameter(config, 'cycles', int, 4)
        self._go_speed = get_value_parameter(config, 'go_speed', float, 15.0)   # km/h
        self._go_time = [_rng.uniform(3.0, 5.0) for _ in range(self._cycles)]
        self._stop_time = [_rng.uniform(2.0, 8.0) for _ in range(self._cycles)]
        self._scenario_timeout = 240
        self._cars = []
        super().__init__("StopAndGo", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable)

    def _move_forward(self, wp, distance):
        dist, nxt = 0.0, wp
        while dist < distance:
            n = nxt.next(1.0)
            if not n:
                break
            nxt = n[0]
            dist += 1.0
        return nxt

    def _initialize_actors(self, config):
        start_wp = self._map.get_waypoint(config.trigger_points[0].location)
        for i in range(self._num_cars):
            wp = self._move_forward(start_wp, self._distance + i * 9.0)
            tf = wp.transform
            tf.location.z += 0.5
            car = CarlaDataProvider.request_new_actor(
                'vehicle.*', tf, rolename='scenario', attribute_filter={'base_type': 'car', 'generation': 2})
            if car is None:
                continue
            car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
            car.apply_control(carla.VehicleControl(hand_brake=True))
            self._cars.append(car)
            self.other_actors.append(car)
        if not self._cars:
            raise ValueError("StopAndGo: could not spawn any car")
        self._end_wp = self._move_forward(start_wp, self._distance + 120)
        print(f">> [StopAndGo] {len(self._cars)} cars from {self._distance:.1f}m, {self._cycles} cycles, "
              f"stops {[round(t, 1) for t in self._stop_time]} s")

    def _car_cycle(self, car):
        seq = py_trees.composites.Sequence()
        seq.add_child(HandBrakeVehicle(car, False))
        for go, stop in zip(self._go_time, self._stop_time):
            seq.add_child(ChangeAutoPilot(car, True, parameters={"auto_lane_change": False, "max_speed": self._go_speed}))
            seq.add_child(Idle(go))
            seq.add_child(ChangeAutoPilot(car, False))
            seq.add_child(StopVehicle(car, 1.0))
            seq.add_child(Idle(stop))
        seq.add_child(ChangeAutoPilot(car, True, parameters={"auto_lane_change": False}))
        seq.add_child(WaitForever())
        return seq

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="StopAndGo")
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 20))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))
        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self._cars[0].get_location(), 15.0))
        cycles = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        for car in self._cars:
            cycles.add_child(self._car_cycle(car))
        behavior.add_child(cycles)
        end_condition.add_child(behavior)
        if self.route_mode:
            end_condition.add_child(ClearEgoLaneAhead(self.ego_vehicles[0], self._cars[0],
                                                      [a.id for a in self.other_actors]))
        root.add_child(end_condition)
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))
        return root

    def _create_test_criteria(self):
        return [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]

    def __del__(self):
        self.remove_all_actors()


class RoadDebris(BasicScenario):
    """Fallen object lying in the ego lane (NOT registered for the expert).
    params: size = small (flat debris, can be driven over) | large (must not be hit)
            prop = exact blueprint id (optional, overrides size)
            distance (m from trigger), offset (lateral m, + = right)
            side_flow = 1 -> endless stream of cars in the lane a bypass would use (busy neighbour lane)
            flow_speed (m/s, default 11 ~ 40 km/h), flow_min / flow_max (m between cars, default 20 / 70)
    Question for the LLM: drive over it (PROCEED), go around (BYPASS) or wait (WAIT)?"""

    CANDIDATES = {
        'small': ['static.prop.dirtdebris01', 'static.prop.dirtdebris02', 'static.prop.plasticbag',
                  'static.prop.creasedbox01', 'static.prop.trashbag'],
        'large': ['static.prop.box03', 'static.prop.box02', 'static.prop.barrel', 'static.prop.ironplank',
                  'static.prop.shoppingtrolley', 'static.prop.box01'],
    }

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._distance = get_value_parameter(config, 'distance', float, 40) + _rng.uniform(-5.0, 5.0)
        self._size = get_value_parameter(config, 'size', str, 'large')
        self._prop_id = get_value_parameter(config, 'prop', str, '')
        self._offset = get_value_parameter(config, 'offset', float, 0.0)
        self._side_flow = get_value_parameter(config, 'side_flow', int, 0) == 1
        self._flow_speed = get_value_parameter(config, 'flow_speed', float, 11.0)
        self._flow_gap = [get_value_parameter(config, 'flow_min', float, 20.0),
                          get_value_parameter(config, 'flow_max', float, 70.0)]
        self._flow = None
        self._scenario_timeout = 240
        super().__init__("RoadDebris", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable)

    def _move_forward(self, wp, distance):
        dist, nxt = 0.0, wp
        while dist < distance:
            n = nxt.next(1.0)
            if not n:
                break
            nxt = n[0]
            dist += 1.0
        return nxt

    def _move_back(self, wp, distance):
        dist, prv = 0.0, wp
        while dist < distance:
            p = prv.previous(1.0)
            if not p:
                break
            prv = p[0]
            dist += 1.0
        return prv

    def _bypass_lane(self, wp):
        """Neighbour lane a bypass would use (same choice as the agent): left same-dir, right same-dir, else
        left opposite. Returns (lane_wp_function, same_direction) or (None, None)."""
        drive = carla.LaneType.Driving
        left, right = wp.get_left_lane(), wp.get_right_lane()
        if left is not None and left.lane_type == drive and left.lane_id * wp.lane_id > 0:
            return (lambda w: w.get_left_lane()), True
        if right is not None and right.lane_type == drive and right.lane_id * wp.lane_id > 0:
            return (lambda w: w.get_right_lane()), True
        if left is not None and left.lane_type == drive:
            return (lambda w: w.get_left_lane()), False
        return None, None

    def _setup_flow(self, debris_wp):
        side, same = self._bypass_lane(debris_wp)
        if side is None:
            print(">> [RoadDebris] side_flow requested but there is no neighbour lane -> no flow")
            return
        behind = side(self._move_back(debris_wp, 90))
        ahead = side(self._move_forward(debris_wp, 60))
        if behind is None or ahead is None:
            print(">> [RoadDebris] could not place the side flow -> no flow")
            return
        src, snk = (behind, ahead) if same else (ahead, behind)
        self._flow = (src, snk)
        print(f">> [RoadDebris] busy neighbour lane: cars every {self._flow_gap[0]:.0f}-{self._flow_gap[1]:.0f} m "
              f"at {self._flow_speed * 3.6:.0f} km/h ({'same' if same else 'opposite'} direction)")

    def _pick_blueprint(self):
        lib = self._world.get_blueprint_library()
        ids = {b.id for b in lib.filter('static.prop.*')}
        for cand in ([self._prop_id] if self._prop_id else []) + self.CANDIDATES.get(self._size, []):
            if cand in ids:
                return cand
        raise ValueError(f"RoadDebris: none of the props for size={self._size} exist in this CARLA build")

    def _initialize_actors(self, config):
        start_wp = self._map.get_waypoint(config.trigger_points[0].location)
        wp = self._move_forward(start_wp, self._distance)
        tf = wp.transform
        r = tf.get_right_vector()
        loc = tf.location + carla.Location(x=r.x * self._offset, y=r.y * self._offset, z=0.05)
        rot = carla.Rotation(yaw=tf.rotation.yaw + _rng.uniform(-40, 40))
        bp_id = self._pick_blueprint()
        prop = CarlaDataProvider.request_new_actor(bp_id, carla.Transform(loc, rot), rolename='scenario')
        if prop is None:
            raise ValueError(f"RoadDebris: could not spawn {bp_id}")
        self._prop = prop
        self.other_actors.append(prop)
        # sit the object ON the road: its pivot is not always at the bottom (some meshes have it in the
        # middle -> half of the object would be under the asphalt). Move it so the bounding-box bottom = road.
        spawn_tf = carla.Transform(loc, rot)
        road_z = tf.location.z          # lane-centre height of the road (a ground ray would hit the object itself)
        bottom = min(v.z for v in prop.bounding_box.get_world_vertices(spawn_tf))
        dz = road_z + 0.01 - bottom
        if abs(dz) > 0.005:
            prop.set_transform(carla.Transform(loc + carla.Location(z=dz), rot))
        # PROP_PHYSICS=1: the object is a movable rigid body (gets pushed / knocked away) instead of an
        # immovable static prop that stops the car like a concrete block
        self._physics = os.environ.get('PROP_PHYSICS', '0') == '1'
        if self._physics:
            prop.set_simulate_physics(True)
        self._end_wp = self._move_forward(wp, 40)
        if self._side_flow:
            self._setup_flow(wp)
        e = prop.bounding_box.extent
        print(f">> [RoadDebris] {bp_id} ({self._size}) {self._distance:.1f}m ahead at "
              f"({loc.x:.1f}, {loc.y:.1f}), size {2*e.x:.2f} x {2*e.y:.2f} x {2*e.z:.2f} m, "
              f"bottom was {bottom - road_z:+.2f} m vs road -> moved {dz:+.2f} m, physics={'on' if self._physics else 'off'}, "
              f"NOT registered for the expert")

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="RoadDebris")
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 10))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))
        if self.route_mode:
            end_condition.add_child(ClearEgoLaneAhead(self.ego_vehicles[0], self._prop, [self._prop.id]))
        if self._flow is not None and SparseFlow is not None:
            flow = SparseFlow(self._flow[0], self._flow[1], self._flow_gap, sink_dist=4,
                             actor_speed=self._flow_speed, initial_actors=True)
            # normal traffic-manager cars (they brake for the ego) instead of the default 'ghost' constant velocity
            flow._is_constant_velocity_active = False
            end_condition.add_child(flow)
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self._prop))
        return root

    def _create_test_criteria(self):
        return [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]

    def __del__(self):
        self.remove_all_actors()


class NarrowCorridor(BasicScenario):
    """Construction zone on a two-way road: traffic cones on the OUTER edges of BOTH lanes leave only a
    temporary corridor straddling the centre line (not in the map). NOT registered for the expert, so
    PDM-Lite keeps following its own lane and hits the cones.
    params: distance (m from trigger to the first cone, default 40), length (default 24 m),
            width (free corridor width, default 3.6 m), spacing (between cones, default 3 m),
            oncoming = 0 (opposite background traffic switched off near the zone)"""

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._distance = get_value_parameter(config, 'distance', float, 40) + _rng.uniform(-4.0, 4.0)
        self._length = get_value_parameter(config, 'length', float, 24)
        self._width = get_value_parameter(config, 'width', float, 3.6)
        self._spacing = get_value_parameter(config, 'spacing', float, 3.0)
        self._oncoming = get_value_parameter(config, 'oncoming', int, 0) == 1
        # layout: centreline (default) = corridor straddling the centre line (go LEFT)
        #         lane   = cones on both edges of OUR lane, corridor = our narrowed lane (go STRAIGHT / center)
        #         closed = rows of cones across both lanes, no way through (WAIT)
        self._layout = get_value_parameter(config, 'layout', str, 'centreline').lower()
        self._offset = get_value_parameter(config, 'offset', float, 0.0)
        # optional sign at the zone entrance (e.g. an arrow board): blueprint, extra yaw (deg), lateral (m, + = right
        # of our lane centre), metres BEFORE the first cone
        self._sign_bp = get_value_parameter(config, 'sign', str, '')
        self._sign_yaw = get_value_parameter(config, 'sign_yaw', float, 0.0)
        self._sign_lat = get_value_parameter(config, 'sign_lat', float, 1.0)
        self._sign_before = get_value_parameter(config, 'sign_before', float, 4.0)
        # arrow drawn on the sign's board (CARLA's arrow board has a blank face): left | right | '' (none)
        self._sign_arrow = get_value_parameter(config, 'sign_arrow', str, '').lower()
        self._arrow_h = get_value_parameter(config, 'arrow_h', float, 2.6)
        self._scenario_timeout = 240
        self._cones = []
        super().__init__("NarrowCorridor", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable)

    def _fwd(self, wp, d):
        n = wp.next(d)
        return n[0] if n else wp

    def _initialize_actors(self, config):
        start_wp = self._map.get_waypoint(config.trigger_points[0].location)
        wp0 = self._fwd(start_wp, self._distance)
        half_lane = wp0.lane_width / 2.0
        s = 0.0
        while s <= self._length + 1e-3:
            wp = self._fwd(wp0, s) if s > 0 else wp0
            tf = wp.transform
            r = tf.get_right_vector()
            # centre line is half a lane to the LEFT of our lane centre; cones at +-(width/2 + 0.2) from it
            if self._layout == 'lane':
                lats = [self._offset + side * (self._width / 2.0 + 0.2) for side in (+1, -1)]
            elif self._layout == 'closure':
                # OUR lane is closed: a short row across the lane at the entrance, then cones along both lane edges
                edge = half_lane - 0.15
                lats = [0.0, edge * 0.7] if s == 0.0 else [-edge, edge]
            elif self._layout == 'closed':
                lats, x = [], -3.0 * half_lane + 0.5
                while x <= half_lane - 0.3:
                    lats.append(x)
                    x += 1.1
            else:
                lats = [-half_lane + side * (self._width / 2.0 + 0.2) for side in (+1, -1)]
            for lat in lats:
                loc = tf.location + carla.Location(x=r.x * lat, y=r.y * lat, z=0.05)
                cone = CarlaDataProvider.request_new_actor('static.prop.constructioncone',
                                                           carla.Transform(loc, tf.rotation), rolename='scenario')
                if cone is not None:
                    cone.set_simulate_physics(False)
                    self._cones.append(cone)
                    self.other_actors.append(cone)
            s += self._spacing
        if not self._cones:
            raise ValueError("NarrowCorridor: could not spawn any cone")
        if self._sign_bp:
            if self._sign_before >= 0:
                prev = wp0.previous(self._sign_before) if self._sign_before > 0 else [wp0]
            else:                                    # negative = AFTER the first cone (inside the closed zone)
                prev = wp0.next(-self._sign_before)
            swp = prev[0] if prev else wp0
            stf = swp.transform
            r = stf.get_right_vector()
            loc = stf.location + carla.Location(x=r.x * self._sign_lat, y=r.y * self._sign_lat, z=0.05)
            rot = carla.Rotation(yaw=stf.rotation.yaw + self._sign_yaw)
            sign = CarlaDataProvider.request_new_actor(self._sign_bp, carla.Transform(loc, rot), rolename='scenario')
            if sign is not None:
                sign.set_simulate_physics(False)
                self.other_actors.append(sign)
                self._sign = sign
                print(f">> [NarrowCorridor] sign {sign.type_id} yaw+{self._sign_yaw:.0f} lat {self._sign_lat:+.1f} m, "
                      f"{self._sign_before:.0f} m before the first cone")
                if self._sign_arrow in ('left', 'right'):
                    # big orange arrow just in front of the board, pointing left/right as seen from the ego car
                    f = stf.get_forward_vector()
                    c = loc + carla.Location(x=-f.x * 0.35, y=-f.y * 0.35, z=self._arrow_h)
                    d = -1.0 if self._sign_arrow == 'left' else 1.0
                    half = 0.65
                    tail = c + carla.Location(x=-r.x * d * half, y=-r.y * d * half)
                    head = c + carla.Location(x=r.x * d * half, y=r.y * d * half)
                    self._world.debug.draw_arrow(tail, head, thickness=0.18, arrow_size=0.45,
                                                 color=carla.Color(255, 120, 0), life_time=0.0)
                    print(f">> [NarrowCorridor] arrow '{self._sign_arrow}' drawn on the sign board (h {self._arrow_h} m)")
            else:
                print(f">> [NarrowCorridor] could not spawn sign {self._sign_bp}")
        self._end_wp = self._fwd(wp0, self._length + 15.0)
        print(f">> [NarrowCorridor] {len(self._cones)} cones, zone {self._distance:.1f}-{self._distance + self._length:.1f} m "
              f"ahead, layout={self._layout} (corridor {self._width:.1f} m), NOT registered for the expert")

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="NarrowCorridor")
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + self._length))
            if not self._oncoming:
                root.add_child(ChangeOppositeBehavior(active=False))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))
        if self.route_mode:
            # background cars ahead in our lane drive into the cones (traffic manager ignores props) and get
            # stuck there -> remove them before they reach the zone. Target = last cone on our side.
            end_condition.add_child(ClearEgoLaneAhead(self.ego_vehicles[0], self._cones[-2],
                                                      [c.id for c in self._cones]))
        root.add_child(end_condition)
        if self.route_mode and not self._oncoming:
            root.add_child(ChangeOppositeBehavior(active=True))
        for c in self._cones:
            root.add_child(ActorDestroy(c))
        if getattr(self, '_sign', None) is not None:
            root.add_child(ActorDestroy(self._sign))
        return root

    def _create_test_criteria(self):
        return [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]

    def __del__(self):
        self.remove_all_actors()
