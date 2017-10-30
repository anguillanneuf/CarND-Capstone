#!/usr/bin/env python

import rospy
import numpy as np
import tf
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from python_common.helper import Math3DHelper

import math
from scipy.interpolate import CubicSpline
import copy
import yaml

"""
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
"""

LOOKAHEAD_WPS = 10  # Number of waypoints we will publish. You can change this number
T_STEP_SIZE = 0.02  # time step for slowdown or speedup
LATENCY = 0.2  # 100ms latency from planner to vehicle /simulator
STALE_TIME = 1
LOG = False  # Set to true to enable logs


class WaypointUpdater(object):
    """ Publishes waypoints from the car's current position to some `x` distance ahead."""

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # for normal driving
        self.waypoints = None
        self.base_waypoints=None
        self.base_waypoints_reverse=None

        # distance to next waypoint
        self.base_wp_dists = []
        self.total_wp_num = 0
        self.current_pose = None
        self.next_wp_idx = 0
        self.current_vel = 0

        # for traffic light handling
        self.tf_state = "no_traffic"
        self.augmented_waypoints = None
        # stoplines position
        self.stop_line_index = None

        # dbw status
        self.dbw_enabled = False
        # car drive direction, +1: along base_waypoints, -1: opposite of base_waypoints
        self.car_dir = 1

        self.target_vel = rospy.get_param('/waypoint_loader/velocity', 40) * 0.27778
        self.max_accel = rospy.get_param('~max_accel', 8.)
        self.max_brake = rospy.get_param('~max_brake', 10.)
        self.max_jerk = rospy.get_param('~max_jerk',10.)

        # Car's position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # Car's velocity
        rospy.Subscriber(
            '/current_velocity',
            TwistStamped,
            self.current_vel_cb)
        # Waypoints to follow (coming from waypoint_loader)
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        # Traffic lights (coming from tge Perception subsystem)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # Obstacles (coming from the Perception subsystem)
        rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)
        # subscribe the dbw_enabled to check car's position and orientation and set correct direction
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Final waypoints (for the control subsystem)
        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            rate.sleep()

            if self.waypoints is None or self.current_pose is None or \
                            self.dbw_enabled is False or self.stop_line_index is None:
                continue

            handlers = {"no_traffic":self.handle_no_traffic,
                        "in_stopping":self.handle_in_stopping}
            handlers[self.tf_state]()

            # get next waypoint index
            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now()
            # find next waypoint index
            predicted_wp,next_wp_idx = WaypointUpdater.predict_next_waypoint(self.waypoints, self.current_pose,
                                                                             self.current_vel, self.next_wp_idx)

            # LOOP OVER FROM THE BEGINNING
            if next_wp_idx == self.total_wp_num:
                self.next_wp_idx = 0
                predicted_wp, next_wp_idx = WaypointUpdater.predict_next_waypoint(self.waypoints, self.current_pose,
                                                                                  self.current_vel, self.next_wp_idx)

            #if next_wp_idx != self.next_wp_idx:
            #    rospy.loginfo("Next WayPoint:%d", next_wp_idx)

            if self.augmented_waypoints is not None:
                predicted_wp,next_wp = WaypointUpdater.predict_next_waypoint(self.augmented_waypoints, self.current_pose,
                                                                             self.current_vel, 0)
                lane.waypoints = self.augmented_waypoints[predicted_wp:]
                self.augmented_waypoints = self.augmented_waypoints[predicted_wp:]
            else:
                # publish normal waypoints
                stop = predicted_wp + LOOKAHEAD_WPS
                if stop < self.total_wp_num:
                    lane.waypoints = self.waypoints[predicted_wp:stop]
                else:
                    stop = stop % self.total_wp_num
                    lane.waypoints = self.waypoints[predicted_wp:] + self.waypoints[:stop]

            self.next_wp_idx = next_wp_idx
            self.final_waypoints_pub.publish(lane)

    def dbw_enabled_cb(self,msg):
        self.dbw_enabled = msg.data
        if self.dbw_enabled is True and self.current_pose is not None and self.base_waypoints is not None:
            # check car heading and set car direction
            self.next_wp_idx = WaypointUpdater.find_nearst_waypoint(self.base_waypoints, self.current_pose)\
                               % self.total_wp_num

            cp = self.current_pose.pose
            # car's heading
            (_, _, car_heading) = tf.transformations.euler_from_quaternion([cp.orientation.x,
                                                                    cp.orientation.y,
                                                                    cp.orientation.z,
                                                                    cp.orientation.w])
            x1 = self.base_waypoints[self.next_wp_idx].pose.pose.position.x
            y1 = self.base_waypoints[self.next_wp_idx].pose.pose.position.y
            x2 = self.base_waypoints[(self.next_wp_idx+1)%self.total_wp_num].pose.pose.position.x
            y2 = self.base_waypoints[(self.next_wp_idx+1)%self.total_wp_num].pose.pose.position.y
            wp_heading = math.atan2(y2 - y1, x2 - x1)

            if math.cos(car_heading-wp_heading) > 0:
                self.car_dir = 1
                self.waypoints = self.base_waypoints
                rospy.loginfo("self-driving..  car runs in waypoints direction, ")
            else:
                self.car_dir = -1
                self.waypoints = self.base_waypoints_reverse
                self.next_wp_idx = WaypointUpdater.find_nearst_waypoint(self.waypoints, self.current_pose) \
                                   % self.total_wp_num
                rospy.loginfo("self-driving..  car runs in opposite direction of waypoints")

    def handle_no_traffic(self):

        if self.stop_line_index != -1:
            rospy.loginfo("Stop line at wp:%d tl RED,state -> IN_STOPPING", self.stop_line_index)
            if self.car_dir == 1:
                real_stopline = (self.stop_line_index -5) %self.total_wp_num
            else:
                real_stopline = (self.total_wp_num - self.stop_line_index -5) % self.total_wp_num

            if real_stopline > self.next_wp_idx:
                d = sum(self.base_wp_dists[self.next_wp_idx:real_stopline])
            else:
                d = sum(self.base_wp_dists[self.next_wp_idx:]) + sum(self.base_wp_dists[:real_stopline])

            self.augmented_waypoints = self.generate_brake_path(d,real_stopline)
            if self.augmented_waypoints is not None:
                self.tf_state = "in_stopping"

    def handle_in_stopping(self):
        if self.stop_line_index == -1:
            # self.augmented_wps = self.generate_speedup_path()
            rospy.loginfo("Traffic Light is  GREEN, car moves")
            self.augmented_waypoints = None
            self.tf_state = "no_traffic"

    def pose_cb(self, msg):
        """Car's position callback"""
        if self.current_pose is None and self.waypoints is not None:
            self.next_wp_idx = WaypointUpdater.find_nearst_waypoint(self.waypoints, msg) % self.total_wp_num

        self.current_pose = msg

    def base_waypoints_cb(self, waypoints):
        """waypoints to follow callback"""
        if self.waypoints is None:
            self.base_waypoints = waypoints.waypoints
            self.base_waypoints_reverse = self.base_waypoints[::-1]

            self.waypoints = self.base_waypoints
            self.total_wp_num = len(self.base_waypoints)
            self.base_wp_dists = [ Math3DHelper.distance(
                self.waypoints[i].pose.pose.position,
                self.waypoints[i + 1].pose.pose.position) for i in range(self.total_wp_num - 1)]
            # last point to first point as a loop
            self.base_wp_dists.append(Math3DHelper.distance(
                self.waypoints[-1].pose.pose.position,
                self.waypoints[0].pose.pose.position))

            if self.current_pose is not None:
                self.next_wp_idx = WaypointUpdater.find_nearst_waypoint(self.waypoints, self.current_pose) \
                                   % self.total_wp_num

    def traffic_cb(self, msg):
        self.stop_line_index = msg.data
        #rospy.logwarn("traffix %s", msg)

    def current_vel_cb(self, msg):
        """Car's velocity"""
        self.current_vel = msg.twist.linear.x

    def obstacle_cb(self, msg):
        """Obstacles detected by the perception subsystem"""
        pass

    def generate_brake_path(self,distance,stop_wp, emergency=False):
        """
        Generate path for braking state
        :param distance: distance required
        :param stop_wp: waypoint index to stop
        :param emergency: emergency brake
        :return: list of waypoints
        """
        # alias
        min_d = WaypointUpdater.get_min_distance

        d_brake_min = min_d(self.current_vel, 0, self.max_brake, self.max_jerk)
        d_speedup = min_d(self.current_vel, self.target_vel, self.max_accel / 2, self.max_jerk)
        d_brake_normal = min_d(self.target_vel, 0.0, self.max_brake / 2, self.max_jerk)
        # check if there is enough distance to brake to stop
        if distance < d_brake_min:
            rospy.logwarn("Underbrake !!!, required d,%.03f, actual. %.03f", d_brake_min, distance)
            if emergency is False:
                return None
            interpolated_wps, start, stop = self.generate_brake_path_with_max_brake(stop_wp)
        # enough distance to speed up and brake normal, normal means 0.5 x max_accel or 0.5 max_brake
        elif distance > d_speedup + d_brake_normal:
            interpolated_wps, start, stop = self.generate_brake_path_with_half_max_brake(stop_wp)
        else:
            interpolated_wps, start, stop = self.generate_brake_path_with_adapted_velocity(stop_wp, distance)

        # append some waypoints with zero velocity
        def zero_speed(p):
            p.twist.twist.linear.x = 0
            return p

        if stop + LOOKAHEAD_WPS > self.total_wp_num:
            additional_wps = copy.deepcopy(self.waypoints[stop:])
            additional_wps += copy.deepcopy(self.waypoints[:(stop + LOOKAHEAD_WPS) % self.total_wp_num])
        else:
            additional_wps = copy.deepcopy(self.waypoints[stop:stop + LOOKAHEAD_WPS])
        additional_wps = map(zero_speed, additional_wps)

        rospy.loginfo("next wp:%d, start:%d,stop:%d, inter:%d,total:%d, distance:%.03f",
                      self.next_wp_idx,start,stop,len(interpolated_wps),self.total_wp_num,distance)

        if start > self.total_wp_num:
            return self.waypoints[self.next_wp_idx:] + self.waypoints[:start % self.total_wp_num]\
                   + list(interpolated_wps) + additional_wps

        return self.waypoints[self.next_wp_idx:start] + list(interpolated_wps) + additional_wps

    def generate_brake_path_with_max_brake(self,stop_wp):
        ds_slowdown, vs_slowdown = WaypointUpdater.generate_dist_vels(
            self.current_vel, 0, self.max_brake, self.max_jerk)
        interpolated_wps, start, stop = WaypointUpdater.interpolate_waypoints(
            self.waypoints,self.base_wp_dists, stop_wp, ds_slowdown[-1], ds_slowdown, vs_slowdown)

        rospy.loginfo("Slow down v from %.03f m/s with d %.03f m", self.current_vel, ds_slowdown[-1])

        return interpolated_wps,start,stop

    def generate_brake_path_with_half_max_brake(self, stop_wp):
        if self.current_vel > 0.8 * self.target_vel:
            ds_speedup = [0]
            vs_speedup = [self.current_vel]
        else:
            ds_speedup, vs_speedup = WaypointUpdater.generate_dist_vels(
                self.current_vel, self.target_vel, self.max_accel / 2, self.max_jerk)
        # trick to set constant velocity
        vs_speedup = [vs_speedup[-1]] * len(vs_speedup)
        ds_slowdown, vs_slowdown = WaypointUpdater.generate_dist_vels(
            vs_speedup[-1], 0, self.max_brake / 2, self.max_jerk)

        ds_slowdown = ds_slowdown + ds_speedup[-1]
        ds_combined = list(ds_speedup) + list(ds_slowdown[1:])
        vs_combined = list(vs_speedup) + list(vs_slowdown[1:])
        interpolated_wps, start, stop = WaypointUpdater.interpolate_waypoints(
            self.waypoints, self.base_wp_dists,stop_wp, ds_combined[-1], ds_combined, vs_combined)

        rospy.loginfo("Speed up from %.03f m/s to %.03f m/s with d %.03f m and slow down with d %.03f m",
                      self.current_vel, vs_speedup[-1], ds_speedup[-1], ds_slowdown[-1]-ds_speedup[-1])

        return interpolated_wps, start, stop

    def generate_brake_path_with_adapted_velocity(self, stop_wp,distance):
        min_d = WaypointUpdater.get_min_distance
        # find a proper velocity
        v0 = self.current_vel
        vt = self.target_vel
        delta_d = -1
        scale = 10
        dv = np.abs(vt - v0)

        while delta_d < 0:
            for i in range(0, scale):
                vt = vt - dv / scale
                d_speedup = min_d(v0, vt, self.max_accel, self.max_jerk)
                d_slowdown = min_d(vt, 0.0, self.max_brake, self.max_jerk)
                delta_d = distance - d_speedup - d_slowdown
                if delta_d > 0:
                    break
            # scaled down the velocity
            dv = dv / scale

        if delta_d < 0:
            rospy.logerr(
                "can not find a proper v to speed up, error in algorithm")
            return None,self.next_wp_idx,self.next_wp_idx

        if v0 > 0.8 * self.target_vel:
            ds_speedup = [0]
            vs_speedup = [v0]
            ds_slowdown, vs_slowdown = WaypointUpdater.generate_dist_vels(
                v0, 0, self.max_brake, self.max_jerk)
            rospy.loginfo("Slow down v from %.03f m/s with d %.03f m", v0, distance)
        else:
            ds_speedup, vs_speedup = WaypointUpdater.generate_dist_vels(
                v0, vt, self.max_accel, self.max_jerk)
            # trick, for speed up set the target velocity and let control
            # handle the smoothness
            vs_speedup = [vs_speedup[-1]] * len(vs_speedup)
            ds_slowdown, vs_slowdown = WaypointUpdater.generate_dist_vels(
                vt, 0, self.max_brake, self.max_jerk)
            rospy.loginfo("Speed up from %.03f m/s to %.03f m/s with d %.03f m and slow down with d %.03f m",
                          v0, vt, ds_speedup[-1], distance - ds_speedup[-1])

        # make brake path longer
        d2 = distance - ds_speedup[-1]
        ds_slowdown = ds_slowdown * d2 / ds_slowdown[-1] + ds_speedup[-1]

        ds_combined = list(ds_speedup) + list(ds_slowdown[1:])
        vs_combined = list(vs_speedup) + list(vs_slowdown[1:])

        interpolated_wps, start, stop = WaypointUpdater.interpolate_waypoints(
            self.waypoints, self.base_wp_dists, stop_wp, ds_combined[-1], ds_combined, vs_combined)
        return interpolated_wps, start, stop

    @staticmethod
    def interpolate_waypoints(
            waypoints,
            wp_distances,
            wp_idx,
            d0,
            distances,
            velocities,
            wp_is_start=False):
        """
        :param waypoints:
        :param wp_distances:
        :param wp_idx:
        :param d0: distance of start or end waypoint of generated trajectory
        :param distances:
        :param velocities:
        :param wp_is_start:
        :return: intepolated waypoints with update of velocities,start_wp,stop_wp
        """
        if len(distances) < 2:
            rospy.logwarn("Interpolating does not have enough points")
            return None, 0, 0

        ds_origs = []
        xs_origs = []
        ys_origs = []
        oz_origs = []
        d = d0
        wp_start = wp_stop =  wp_idx

        total_wp = len(waypoints)

        if wp_is_start:
            for i in range(wp_idx, wp_idx + total_wp):
                ds_origs.append(d)
                index = i%total_wp
                xs_origs.append(waypoints[index].pose.pose.position.x)
                ys_origs.append(waypoints[index].pose.pose.position.y)
                oz_origs.append(waypoints[index].pose.pose.orientation.z)
                if d > distances[-1]:
                    wp_stop = i
                    break
                d += wp_distances[index]

        else:
            for i in range(wp_idx, wp_idx - total_wp, -1):
                ds_origs.append(d)
                index = i % total_wp
                xs_origs.append(waypoints[index].pose.pose.position.x)
                ys_origs.append(waypoints[index].pose.pose.position.y)
                oz_origs.append(waypoints[index].pose.pose.orientation.z)
                if d < distances[0]:
                    wp_start = i
                    break
                d -= wp_distances[(index-1) % total_wp]
            # reverse the list for spline interpolation
            ds_origs = ds_origs[::-1]
            xs_origs = xs_origs[::-1]
            ys_origs = ys_origs[::-1]
            oz_origs = oz_origs[::-1]

        if len(ds_origs) < 2:
            rospy.logwarn("Interpolating does not have enough waypoints, target wp:%d",wp_idx)
            return None, 0, 0

        cs_x = CubicSpline(ds_origs, xs_origs, bc_type='natural')
        cs_y = CubicSpline(ds_origs, ys_origs, bc_type='natural')
        cs_oz = CubicSpline(ds_origs, oz_origs, bc_type='natural')

        xs_samples = cs_x(distances)
        ys_samples = cs_y(distances)
        oz_samples = cs_oz(distances)

        interpolated_wps = []
        for i in range(len(distances)):
            p = Waypoint()
            p.pose.pose.position.x = xs_samples[i]
            p.pose.pose.position.y = ys_samples[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = oz_samples[i]
            p.twist.twist.linear.x = velocities[i]
            interpolated_wps.append(p)

        return interpolated_wps, wp_start, wp_stop

    @staticmethod
    def find_nearst_waypoint(waypoints, pose):
        """
        compare the geometric distance
        :param waypoints:
        :param pose:
        :return:
        """
        d2func = WaypointUpdater.distance_2D_square
        ds = [d2func(pose.pose.position,waypoints[i].pose.pose.position) for i in range(len(waypoints))]
        return np.argmin(ds)

    @staticmethod
    def find_next_waypoint(waypoints, pose, start_wp=0):
        """
        Get the next waypoint index
        :param pose: car's position
        :param start_wp: start waypoint index for search
        :return: index of next waypoint
        """
        d_min = float('inf')
        next_wp = start_wp

        for i in range(start_wp, len(waypoints)):
            # only for comparision not necessary to calculate square root .
            d = WaypointUpdater.distance_2D_square(
                pose.pose.position, waypoints[i].pose.pose.position)
            next_wp = i
            if d < d_min:
                d_min = d
            else:
                # calculate angle between two vectors v1=x1 + y1*i, v2= x2 +
                # y2*i
                x1 = waypoints[i].pose.pose.position.x - \
                    waypoints[i - 1].pose.pose.position.x
                y1 = waypoints[i].pose.pose.position.y - \
                    waypoints[i - 1].pose.pose.position.y
                x2 = pose.pose.position.x - \
                    waypoints[i - 1].pose.pose.position.x
                y2 = pose.pose.position.y - \
                    waypoints[i - 1].pose.pose.position.y
                # we only need to check if cos_theta sign to determin the angle
                # is >90
                cos_theta_sign = x1 * x2 + y1 * y2

                if cos_theta_sign < 0:
                    next_wp = i - 1
                # stop search, find the right one
                break

        # check if reaches the last wp.
        if next_wp == len(waypoints) - 1 and len(waypoints) >1:
            x1 = waypoints[-2].pose.pose.position.x - \
                waypoints[-1].pose.pose.position.x
            y1 = waypoints[-2].pose.pose.position.y - \
                waypoints[-1].pose.pose.position.y
            x2 = pose.pose.position.x - waypoints[-1].pose.pose.position.x
            y2 = pose.pose.position.y - waypoints[-1].pose.pose.position.y
            # we only need to check if cos_theta sign to determin the angle is
            # >90
            cos_theta_sign = x1 * x2 + y1 * y2
            if cos_theta_sign < 0:
                next_wp = next_wp + 1

        return next_wp

    @staticmethod
    def predict_next_waypoint(waypoints, pose, vel, start_wp=0, delay=LATENCY):
        """
        Get the next waypoint index
        :param pose: car's position
        :param vel: car's velocity
        :param start_wp: start waypoint index for search
        :param delay: system latency
        :return: predicted waypoint, next waypoint
        # TODO: the algorithmn is not accurate when waypoints are forming a ring
        """
        next_wp = WaypointUpdater.find_next_waypoint(waypoints, pose, start_wp)
        if next_wp >= len(waypoints):
            return next_wp,next_wp

        d0 = Math3DHelper.distance(
            pose.pose.position,
            waypoints[next_wp].pose.pose.position)
        predict_d = vel * delay
        predict_wp = next_wp

        if predict_d > d0:
            d = d0
            for i in range(next_wp + 1, len(waypoints)):
                d += WaypointUpdater.distance_waypoints(waypoints, i - 1, i)
                predict_wp = i
                if d > predict_d:
                    break

        return predict_wp, next_wp

    @staticmethod
    def generate_dist_vels(v0, vt, accel, jerk):
        """
        Use a cubic spline to fit the distance vs speed
        :param v0: start velocity
        :param vt: target velocity
        :param accel: maximum acceleration
        :param jerk:  maximum jerk
        :return: list of calculated distances, list of calculated velocities
        """
        ds, ts = WaypointUpdater.get_min_distance(
            v0, vt, accel, jerk, return_list=True)
        if len(ds) < 2:
            return [0], [v0]

        cs_d = CubicSpline(ts, ds, bc_type='natural')
        cs_v = cs_d.derivative(nu=1)

        # generate sampled points from spline
        ts_samples = np.arange(T_STEP_SIZE, ts[-1] + T_STEP_SIZE, T_STEP_SIZE)
        ds_samples = cs_d(ts_samples)
        vs_samples = cs_v(ts_samples)

        if LOG:
            accel_s = cs_d(ts_samples, 2)
            js = cs_d(ts_samples, 3)
            rospy.loginfo(
                "max accel %.03f, jerk %.03f", np.max(
                    np.abs(accel_s)), np.max(
                    np.abs(js)))

        return ds_samples, vs_samples

    @staticmethod
    def get_min_distance(v0, vt, accel, jerk, return_list=False):
        """
        Calculate min distance needed to reach target velocity, applying max acceleration and jerk
        :param v0: start velocity
        :param vt: target velocity
        :param accel: max acceleration
        :param jerk: max jerk
        :return: list of key distances and list of time points
        """
        is_accel = True
        v0 = float(v0)
        vt = float(vt)
        accel = float(accel)
        jerk = float(jerk)

        if v0 > vt:
            is_accel = False
            # switch the target velocity
            v = v0
            v0 = vt
            vt = v
        # To small to process, return empty list
        dv = vt - v0
        if dv < 0.0001:
            if return_list:
                return [0], [v0]
            else:
                return 0

        tj = accel / jerk
        vj = jerk * tj ** 2 / 2

        # we assume a0 = 0
        if dv <= vj * 2:
            #  acceleration will not reach max value
            t1 = math.sqrt(dv / jerk)
            v1 = v0 + jerk * t1 ** 2 / 2
            a1 = jerk * t1
            d1 = v0 * t1 + jerk * t1 ** 3 / 6

            t2 = t1
            d2 = v1 * t2 + a1 * t2 ** 2 / 2 - jerk * t2 ** 3 / 6

            ts = [0, t1, t1 + t2]
            if is_accel:
                ds = [0, d1, d1 + d2]
            else:
                ds = [0, d2, d2 + d1]
        else:
            # acceleration increases to max value, holds for t2 and decrease to
            # 0
            t1 = tj
            v1 = vj + v0
            d1 = v0 * t1 + jerk * t1 ** 3 / 6

            t2 = (dv - jerk * t1 ** 2) / accel
            d2 = v1 * t2 + accel * t2 ** 2 / 2
            v2 = v1 + accel * t2

            t3 = t1
            d3 = v2 * t3 + accel * t3 ** 2 / 2 - jerk * t3 ** 3 / 6

            ts = [0, t1, t1 + t2, t1 + t2 + t3]
            if is_accel:
                ds = [0, d1, d1 + d2, d1 + d2 + d3]
            else:
                ds = [0, d3, d3 + d2, d3 + d2 + d1]

        if return_list:
            return ds, ts
        else:
            return ds[-1]

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    @staticmethod
    def distance_waypoints(waypoints, start_wp_index, end_wp_index):
        dist = 0
        last_wp_index = start_wp_index
        for current_wp_index in range(start_wp_index, end_wp_index + 1):
            dist += Math3DHelper.distance(
                waypoints[last_wp_index].pose.pose.position,
                waypoints[current_wp_index].pose.pose.position)
            last_wp_index = current_wp_index
        return dist

    @staticmethod
    def distance_2D_square(pos1, pos2):
        return (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
