#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
import yaml
import sys
import numpy as np

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
    ds = [get_square_gap(pose, waypoints[i]) for i in range(len(waypoints))]
    return np.argmin(ds)

def find_closest_waypoint_forwards(waypoints,pose,start_wp=0):
    dmin = float('inf')
    total_wp = len(waypoints)
    for wp in range(start_wp,start_wp+total_wp):
        d = get_square_gap(pose,waypoints[wp%total_wp])
        if d<dmin:
            dmin = d
        else:
            return (wp -1)% total_wp
    return start_wp

class Detector(object):
    def __init__(self):
        self.car_index = None
        self.waypoints = None
        self.stop_wps = []
        rospy.init_node('tl_detector')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

    def pose_cb(self, msg):
        if self.waypoints is None:
            return

        # only first time to search whole waypoints
        if self.car_index is None:
            self.car_index = get_closest_waypoint_index(msg.pose.position, self.waypoints)
            return

        self.car_index = find_closest_waypoint_forwards(self.waypoints,msg.pose.position,self.car_index)

    def waypoints_cb(self, msg):
        self.waypoints = [x.pose.pose.position for x in msg.waypoints]
        self.car_index = None
        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        stop_positions = config['stop_line_positions']
        p1 = Point(stop_positions[0][0],stop_positions[0][1])
        closest_wp = get_closest_waypoint_index(p1,self.waypoints)
        self.stop_wps = []
        for stop in stop_positions:
            p = Point(stop[0],stop[1])
            closest_wp = find_closest_waypoint_forwards(self.waypoints,p,closest_wp)
            self.stop_wps.append(closest_wp)
        self.stop_wps.sort()
