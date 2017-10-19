#!/usr/bin/env python
import rospy
from styx_msgs.msg import TrafficLightArray, TrafficLight
import sys

from detector import Detector, get_closest_waypoint_index,find_closest_waypoint_forwards

class DummyDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.upcoming_red_light_pub.publish(self.last_wp)
        rospy.spin()

    def traffic_cb(self, msg):
        if self.waypoints is None or self.car_index is None:
            return

        # find next stop line
        next_light_index = 0
        for i in range(len(self.stop_wps)):
            if self.stop_wps[i] > self.car_index:
                next_light_index = i
                break

        if msg.lights[next_light_index].state == TrafficLight.GREEN:
            wp = -1
        elif msg.lights[next_light_index].state < TrafficLight.GREEN:
            wp = self.stop_wps[next_light_index]

        self.upcoming_red_light_pub.publish(wp)

