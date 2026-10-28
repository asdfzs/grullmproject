#!/usr/bin/env python
"""
BrokenDownCar scenario (added for the GRU + LLM deadlock project).

Same physical scene as ParkedObstacle (route_obstacles.py):
  - a car stopped in the ego lane, shifted toward one side
  - hazard lights (left + right blinkers) on, hand brake engaged
  - a warning triangle prop placed beside it

The ONLY difference: this scenario does NOT register itself in
CarlaDataProvider.active_scenarios. PDM-Lite (team_code/autopilot.py) reads that
list to know "which scenario is happening, which actor blocks the road, which way
to bypass". Without that entry, PDM-Lite has no bypass plan and only sees an
ordinary stopped vehicle ahead -> expected to wait behind it (deadlock), even
though the side lane is free.

XML usage (same parameters as ParkedObstacle):
  <scenario name="BrokenDownCar_1" type="BrokenDownCar">
     <distance value="40"/>          meters from trigger point to the car
     <direction value="right"/>      side the car leans to
     <speed value="40"/>
     <trigger_point x=".." y=".." yaw=".." z=".."/>
  </scenario>
"""

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.route_obstacles import ParkedObstacle
from srunner.tools.lane_clear import ClearEgoLaneAhead


class BrokenDownCar(ParkedObstacle):

    def _create_behavior(self):
        """ParkedObstacle behavior + keep background cars out of the gap between ego and the car."""
        root = super()._create_behavior()
        if not self.route_mode:
            return root
        cars = [a for a in self.other_actors if a.type_id.startswith('vehicle.')]
        if not cars:
            return root
        par = py_trees.composites.Parallel(name="BrokenDownCarWithClear",
                                           policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        par.add_child(root)
        par.add_child(ClearEgoLaneAhead(self.ego_vehicles[0], cars[0], [a.id for a in self.other_actors]))
        return par

    def _initialize_actors(self, config):
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # warning triangle next to the car (as a real broken-down car would have)
        self._spawn_side_prop(self._starting_wp)

        # the broken-down car itself: hazard lights + hand brake
        self._vehicle_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        broken_actor = self._spawn_obstacle(self._vehicle_wp, 'vehicle.*')

        lights = broken_actor.get_light_state()
        lights |= self._lights
        broken_actor.set_light_state(carla.VehicleLightState(lights))
        broken_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(broken_actor)

        self._end_wp = self._move_waypoint_forward(self._vehicle_wp, self._end_distance)

        # NOTE: intentionally NOT calling
        #   CarlaDataProvider.active_scenarios.append(...)
        # so the expert (PDM-Lite) is not told about this obstacle.
        print(f">> [BrokenDownCar] spawned {broken_actor.type_id} (id={broken_actor.id}) "
              f"{self._distance:.0f}m ahead, direction={self._direction}, NOT registered for the expert")
