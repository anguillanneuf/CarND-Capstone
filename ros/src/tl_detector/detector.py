#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
import yaml
import sys

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

STALE_TIME = 2.0

def get_square_gap(a, b):
    """Returns squared euclidean distance between two 2D points"""
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy

def get_closest_waypoint_index(pose, waypoints):
    """Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
    Args:
        pose (Pose): position to match a waypoint to

    Returns:
        int: index of the closest waypoint in waypoints

    """
    best_gap = float('inf')
    best_index = 0

    for i, waypoint in enumerate(waypoints):
        gap = get_square_gap(pose, waypoint)
        if gap < best_gap:
            best_index, best_gap = i, gap

    return best_index


class Detector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.stop_line_positions = [Point(x, y) for x, y in config['stop_line_positions']]

        self.car_index = None
        self.waypoints = None
        self.stop_map = {}
        self.best_stop_line_index = None
        self.time_received = 0

    def loop(self):
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            rate.sleep()

            if (rospy.get_time() - self.time_received) > STALE_TIME:
                continue

            if self.best_stop_line_index is not None:
                self.upcoming_red_light_pub.publish(self.best_stop_line_index)

    def pose_cb(self, msg):
        if self.waypoints is None:
            return

        self.car_index = get_closest_waypoint_index(msg.pose.position, self.waypoints)

    def waypoints_cb(self, msg):
        if self.waypoints is None:
            self.waypoints = [x.pose.pose.position for x in msg.waypoints]
            for stop in self.stop_line_positions:
                stop_index = get_closest_waypoint_index(stop, self.waypoints)
                self.stop_map[stop] = stop_index
