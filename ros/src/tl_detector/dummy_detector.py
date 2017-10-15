#!/usr/bin/env python
import rospy
from styx_msgs.msg import TrafficLightArray, TrafficLight
import sys

from detector import Detector, get_closest_waypoint_index

class DummyDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.tl_map = {}
        self.tl_map_filled = False

    def traffic_cb(self, msg):
        if self.waypoints is None or self.car_index is None:
            return

        if not self.tl_map_filled:
            lights = [x.pose.pose.position for x in msg.lights]
            for light in lights:
                ind = get_closest_waypoint_index(light, self.stop_line_positions)
                self.tl_map[(light.x, light.y)] = self.stop_line_positions[ind]
            self.tl_map_filled = True

        best_stop_line_index = sys.maxint

        for light in msg.lights:
            if light.state == TrafficLight.GREEN:
                continue

            p = light.pose.pose.position
            stop_line_index = self.stop_map[self.tl_map[(p.x, p.y)]]

            if stop_line_index > self.car_index and \
               stop_line_index < best_stop_line_index:
                best_stop_line_index = stop_line_index

        if best_stop_line_index == sys.maxint:
            self.best_stop_line_index = None
        else:
            self.best_stop_line_index = best_stop_line_index
            self.time_received = rospy.get_time()
