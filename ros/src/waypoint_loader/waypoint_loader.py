#!/usr/bin/env python

import os
import csv
import math

from geometry_msgs.msg import Quaternion

from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

import tf
import rospy
import numpy as np

CSV_HEADER = ['x', 'y', 'z', 'yaw']
MAX_DECEL = 1.0


class WaypointLoader(object):

    def __init__(self):
        rospy.init_node('waypoint_loader', log_level=rospy.DEBUG)
        self.waypoints = None
        self.current_pose = None
        self.car_dir = 1

        # subscribe the dbw_enabled to check car's position and orientation and set correct direction
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        # Car's position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        self.pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1, latch=True)

        self.velocity = self.kmph2mps(rospy.get_param('~velocity'))
        self.new_waypoint_loader(rospy.get_param('~path'))
        rospy.spin()

    def pose_cb(self,msg):
        self.current_pose = msg

    def dbw_enabled_cb(self,msg):
        if msg.data == True and self.current_pose is not None:
            cp = self.current_pose.pose
            total_wp_num  = len(self.waypoints)
            d2func = lambda pos1,pos2: (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2
            ds = [d2func(cp.position, self.waypoints[i].pose.pose.position) for i in range(len(self.waypoints))]
            closed_wp =  np.argmin(ds)
            # check car heading and set car direction
            (_, _, car_heading) = tf.transformations.euler_from_quaternion([cp.orientation.x,
                                                                            cp.orientation.y,
                                                                            cp.orientation.z,
                                                                            cp.orientation.w])
            x1 = self.waypoints[closed_wp].pose.pose.position.x
            y1 = self.waypoints[closed_wp].pose.pose.position.y
            x2 = self.waypoints[(closed_wp + 1) % total_wp_num].pose.pose.position.x
            y2 = self.waypoints[(closed_wp + 1) % total_wp_num].pose.pose.position.y
            wp_heading = math.atan2(y2 - y1, x2 - x1)

            if math.cos(car_heading - wp_heading) > 0:
                car_dir = 1
                rospy.loginfo("self-driving..  car runs in waypoints direction, ")
            else:
                car_dir = -1
                rospy.loginfo("self-driving..  car runs in opposite direction of waypoints")

            if car_dir != self.car_dir:
                self.car_dir = car_dir
                if car_dir == -1:
                    self.publish(self.waypoints[::-1])
                else:
                    self.publish(self.waypoints)


    def new_waypoint_loader(self, path):
        if os.path.isfile(path):
            waypoints = self.load_waypoints(path)
            self.publish(waypoints)
            rospy.loginfo('Waypoint Loded')
        else:
            rospy.logerr('%s is not a file', path)

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def load_waypoints(self, fname):
        waypoints = []
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            for wp in reader:
                p = Waypoint()
                p.pose.pose.position.x = float(wp['x'])
                p.pose.pose.position.y = float(wp['y'])
                p.pose.pose.position.z = float(wp['z'])
                q = self.quaternion_from_yaw(float(wp['yaw']))
                p.pose.pose.orientation = Quaternion(*q)
                p.twist.twist.linear.x = float(self.velocity)

                waypoints.append(p)

        self.waypoints = waypoints
        return waypoints

    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def publish(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointLoader()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint node.')