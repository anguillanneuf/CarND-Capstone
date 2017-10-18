#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from styx_msgs.msg import TrafficLight
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import sys

from detector import Detector

STATE_COUNT_THRESHOLD = 4

class RealDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        self.camera_image = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        rospy.Subscriber('/image_color', Image, self.image_cb)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        if self.state != state:
            self.state_count = 0
            self.state = state
            self.best_stop_line_index = None
        elif self.state_count >= STATE_COUNT_THRESHOLD and self.last_state != self.state:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else None
            self.last_wp = light_wp
            self.best_stop_line_index = light_wp
        else:
            self.best_stop_line_index = self.last_wp
        self.time_received = rospy.get_time()

        self.state_count += 1

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        rgb_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(rgb_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_line_index = self.get_closest_stop_line();
        if closest_line_index is None:
            return -1, TrafficLight.UNKNOWN

        state = self.get_light_state(closest_line_index)

        return closest_line_index, state

    def get_closest_stop_line(self):
        if self.waypoints is None or self.car_index is None:
            return None

        closest_line_index = sys.maxint

        for point, stop_line_index in self.stop_map.iteritems():
            if stop_line_index > self.car_index and \
               stop_line_index < closest_line_index:
                closest_line_index = stop_line_index

        if closest_line_index == sys.maxint: 
            return None
        else:
            return closest_line_index
