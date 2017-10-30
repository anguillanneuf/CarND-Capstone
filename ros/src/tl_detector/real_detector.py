#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from styx_msgs.msg import TrafficLight
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from light_classification.tl_classifier import TLClassifier
import math
import sys

from detector import Detector

STATE_COUNT_THRESHOLD = 4
DETECTION_RANGE = 200

class RealDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.dbw_enabled = False

        # the number of waypoints car brake but over the stopline
        self.brake_buffer = rospy.get_param('~brake_buffer', 5)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=5760000)
        # subscribe the dbw_enabled to check car's position and orientation and set correct direction
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        self.upcoming_red_light_pub.publish(-1)
        rospy.spin()

    def dbw_enabled_cb(self,msg):
        self.dbw_enabled = msg.data

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.car_index is None or self.dbw_enabled is False:
            return

        wp = self.get_closest_stop_line()
        # if the next traffic light is far away, 200 waypoints is 100 meter

        if (wp + 5 - self.car_index)%len(self.waypoints) > DETECTION_RANGE:
            self.upcoming_red_light_pub.publish(-1)
            return

        state = self.process_traffic_lights(msg)

        if self.state != state:
            self.state_count = 1
            self.state = state
        else:
            self.state_count += 1
            if self.state_count >= STATE_COUNT_THRESHOLD:
                if self.state == TrafficLight.GREEN or self.state == TrafficLight.UNKNOWN:
                    wp = -1
                if self.state != TrafficLight.UNKNOWN:
                    self.upcoming_red_light_pub.publish(wp)

    def process_traffic_lights(self,image):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        rgb_image = self.bridge.imgmsg_to_cv2(image, "rgb8")
        return self.light_classifier.get_classification(rgb_image)

    def get_closest_stop_line(self):
        if self.waypoints is None or self.car_index is None:
            return -1

        # find next stop line
        next_stop_index = 0
        for i in range(len(self.stop_wps)):
            if self.car_index < self.stop_wps[i] or\
                 (self.car_index - self.stop_wps[i])%len(self.waypoints) <  self.brake_buffer:
                next_stop_index = i
                break
        return self.stop_wps[next_stop_index]
