#!/usr/bin/env python

import rospy
import numpy as np
import tf
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint,TrafficLightArray
from geometry_msgs.msg import TwistStamped
from python_common.helper import MathHelper
from enum import Enum

import math
from scipy.interpolate import CubicSpline

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 10 # Number of waypoints we will publish. You can change this number
T_STEP_SIZE = 0.02 #time step for slowdown or speedup
LATENCY = 0.2 # 100ms latency from planner to vehicle /simulator
LOG = False # Set to true to enable logs

class Traffic(Enum):

    FREE =1
    IN_BRAKE_ZONE = 2
    IN_STOPPING =3
    SPEED_UP=4

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_vel_cb)
        rospy.Subscriber('/vehicle/traffic_lights',TrafficLightArray,self.traffic_lights_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32,self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.target_vel = rospy.get_param('/waypoint_loader/velocity', 20) * 0.27778
        self.accelerate_rate = rospy.get_param('~accelerate_rate', 1.)
        self.brake_rate = rospy.get_param('~brake_rate', 10.)
        self.max_jerk = rospy.get_param('~max_jerk')
        # only for test in simulator
        self.light_positions = rospy.get_param('~light_positions')
        self.light_pos_wps =[]


        # for normal driving
        self.waypoints = None
        self.total_wp_num = 0
        self.current_pose = None
        self.next_wp_idx = 0
        self.current_vel = 0

        # for traffic light handling
        self.augmented_wps = None

        self.next_tf_idx = -1
        self.check_tf_wp = 0

        self.tl_waypoint= -1
        self.traffic_state = Traffic.FREE

        rospy.spin()

    def generate_brake_path(self, stop_wp,distance):
        '''
        :param stop_wp: waypoint index where car stops
        :param distance: distance required
        :return: list of waypoints
        '''

        v0 = self.current_vel
        vt = self.target_vel
        accel = self.accelerate_rate/2
        brake = self.brake_rate /2

        if v0 < vt:
            d_speedup = WaypointUpdater.get_min_distance(v0,vt, accel, self.max_jerk)
        else:
            d_speedup = 0

        d_slowdown= WaypointUpdater.get_min_distance(vt,0, brake, self.max_jerk)
        delta_d = distance - d_speedup - d_slowdown
        # car has long enough distance to speed up first then slow down
        if delta_d > 0:
            # generate the path for slow down only. speed up path will not cross over the brake down path
            return self.generate_brake_path_imp(stop_wp,vt,brake,self.max_jerk)
        elif distance < WaypointUpdater.get_min_distance(0, self.current_vel, self.brake_rate, self.max_jerk):
            # not enough distance to brake to stop
            d = WaypointUpdater.get_min_distance(0, self.current_vel, self.brake_rate, self.max_jerk)
            rospy.logwarn("Not enough distance for brake, required,%.03f, actual. %.03f",d,distance)
            return None
        else:
            # we can speedup shortly and then slowdown,
            # what is the proper velocity to speed up? we sample from max target velocity to current velocity
            dv = vt - v0
            scale = 5
            while delta_d <0:
                for i in range(1,scale):
                    vt = vt - dv/scale
                    d_speedup = WaypointUpdater.get_min_distance(v0, vt, self.accelerate_rate, self.max_jerk)
                    d_slowdown = WaypointUpdater.get_min_distance(0, vt, self.brake_rate, self.max_jerk)

                    delta_d = distance - d_speedup - d_slowdown
                    if delta_d >0:
                        break
                # scaled down the velocity
                dv = dv/scale

            if delta_d < 0:
                rospy.logerr("can not find a proper v to speed up")
                return self.generate_brake_path_imp(stop_wp, v0, self.brake_rate, self.max_jerk)
            else:
                rospy.logwarn("Speed up and slow down find proper v0:%.03f,vt: %.03f", v0,vt)
                ds_speedup, vs_speedup = WaypointUpdater.generate_dist_vels(v0, vt, self.accelerate_rate, self.max_jerk)
                ds_slowdown, vs_slowdown = WaypointUpdater.generate_dist_vels(vt, 0, self.brake_rate, self.max_jerk)
                interpolated_wps, start, stop = WaypointUpdater.interpolate_waypoints(self.waypoints, stop_wp, ds_slowdown[-1],
                                                                                      ds_slowdown, vs_slowdown,
                                                                                      wp_is_start=False)

                d2 = ds_slowdown[-1] + delta_d
                ds_slowdown = ds_slowdown * d2/ds_slowdown[-1] + ds_speedup[-1]

                ds_combined = list(ds_speedup) + list(ds_slowdown[1:])
                vs_combined = list(vs_speedup) + list(vs_slowdown[1:])

                d0 = ds_combined[-1]
                interpolated_wps,start,stop = WaypointUpdater.interpolate_waypoints(self.waypoints,stop_wp,d0,
                                                ds_combined,vs_combined,wp_is_start=False)

                additional_wps = []
                for wp in self.waypoints[stop:stop+10]:
                    p=Waypoint()
                    p.pose.pose.position.x = wp.pose.pose.position.x
                    p.pose.pose.position.y = wp.pose.pose.position.y
                    p.pose.pose.position.z = wp.pose.pose.position.z
                    p.pose.pose.orientation.z = wp.pose.pose.orientation.z
                    p.twist.twist.linear.x = 0
                    additional_wps.append(p)

                return list(interpolated_wps)+ additional_wps

    def checkwp_before_traffic_light(self, traffic_wp):
        '''
        :param traffic_wp:
        :return: return a wp car needs to check if traffic light color
        '''
        d_min = WaypointUpdater.get_min_distance(self.target_vel,0.0,self.brake_rate/2,
                                                 self.max_jerk,return_list=False)
        d = 0
        check_wp = 0
        for i in range(traffic_wp,0,-1):
            d += WaypointUpdater.distance(self.waypoints,i-1,i)
            if d > d_min:
                check_wp = i-5 # reserve 4 waypoints before enter the brake zone
                break

        return check_wp

    def generate_brake_path_imp(self,stop_wp, v0, deaccel, jerk):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param stop_wp: waypoint index of position where car stops
        :param v0: speed it starts to brake.
        :param deaccel: dacceleration rate
        :param jerk:
        :return: list of waypoints, list of velocity
        '''
        ds_samples,vs_samples = WaypointUpdater.generate_dist_vels(v0,0.0,deaccel,jerk)

        if len(ds_samples) < 2:
            return [], -1

        d0 = ds_samples[-1]
        interpolated_wps,_,_ = \
            WaypointUpdater.interpolate_waypoints(self.waypoints, stop_wp, d0, ds_samples, vs_samples, wp_is_start=False)

        # add additional waypoints and set velocitiy=0, this is used when brake waypoints are published,
        # but car still not stops.
        additional_wps = self.waypoints[stop_wp+1:stop_wp + 11]
        for i in range(len(additional_wps)):
            p = Waypoint()
            p.pose.pose.position.x = additional_wps[i].pose.pose.position.x
            p.pose.pose.position.y = additional_wps[i].pose.pose.position.y
            p.pose.pose.position.z = additional_wps[i].pose.pose.position.z
            p.pose.pose.orientation.z = additional_wps[i].pose.pose.orientation.z
            p.twist.twist.linear.x = 0

            interpolated_wps.append(p)

        return interpolated_wps

    def generate_speedup_path(self):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :return: interpolated waypoints
        '''

        pose = self.current_pose
        v0 = self.current_vel
        vt = self.target_vel
        accel = self.accelerate_rate
        jerk = self.max_jerk

        next_wp = WaypointUpdater.find_next_waypoint(self.waypoints,pose,self.next_wp_idx)
        d0 = WaypointUpdater.direct_distance(pose.pose.position, self.waypoints[next_wp].pose.pose.position)

        ds_samples, vs_samples = WaypointUpdater.generate_dist_vels(v0, vt, accel, jerk)

        if len(ds_samples) < 2:
            return [], -1

        interpolated_wps,_,stop_wp= \
            WaypointUpdater.interpolate_waypoints(self.waypoints, next_wp, d0, ds_samples, vs_samples, wp_is_start=True)

        # add additional waypoints from base waypoints.
        additional_wps = self.waypoints[stop_wp + 1:stop_wp + 11]
        for i in range(len(additional_wps)):
            p = Waypoint()
            p.pose.pose.position.x = additional_wps[i].pose.pose.position.x
            p.pose.pose.position.y = additional_wps[i].pose.pose.position.y
            p.pose.pose.position.z = additional_wps[i].pose.pose.position.z
            p.pose.pose.orientation.z = additional_wps[i].pose.pose.orientation.z
            p.twist.twist.linear.x = additional_wps[i].twist.twist.linear.x
            interpolated_wps.append(p)

        return interpolated_wps

    def pose_cb(self, msg):
        # TODO: Done Implement

        if self.waypoints is None or self.next_wp_idx >= self.total_wp_num:
            return
        self.current_pose = msg
        # get next waypoint index
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        # find next waypoint index
        predicted_wp,next_wp_idx = WaypointUpdater.predict_next_waypoint(self.waypoints, self.current_pose,
                                                                         self.current_vel,self.next_wp_idx)

        if next_wp_idx > self.next_wp_idx:
            rospy.logwarn("Next WayPoint:%d", next_wp_idx)

        if self.augmented_wps is not None:
            predict_wp,next_wp = WaypointUpdater.predict_next_waypoint(self.augmented_wps,self.current_pose,
                                                                       self.current_vel,0)
            lane.waypoints = self.augmented_wps[predict_wp:]
            self.augmented_wps = self.augmented_wps[predict_wp:]
        else:
            # publish normal waypoints
            start = min(len(self.waypoints) - 1, predicted_wp)
            stop = min(predicted_wp + LOOKAHEAD_WPS, self.total_wp_num)
            lane.waypoints = self.waypoints[start:stop]

        self.next_wp_idx = next_wp_idx
        self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, waypoints):
        # TODO: Done Implement
        # rospy.logwarn('waypoints Received - count:%d',len(waypoints.waypoints))
        if self.waypoints is None:
            self.waypoints = waypoints.waypoints
            self.total_wp_num = len(self.waypoints)
            self.brake_start_wp = self.total_wp_num
            self.check_tf_wp = self.total_wp_num
            # following code is used for test traffic lights
            next_wp = 0
            for tf in self.light_positions:
                p = PoseStamped()
                p.pose.position.x = tf[0]
                p.pose.position.y = tf[1]
                next_wp = WaypointUpdater.find_next_waypoint(self.waypoints,p,next_wp)
                self.light_pos_wps.append(next_wp-2)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement

        if self.waypoints is None or self.current_pose is None:
            return

        return
        # traffic light state = red is detected
        if self.tl_waypoint < msg.data and msg.data >self.next_wp_idx:
            rospy.logwarn("traffic light at Waypoint:%d is RED", self.tl_waypoint)
            self.tl_waypoint = msg.data
            d = self.distance(self.waypoints, self.next_wp_idx, self.tl_waypoint)
            self.brake_wps, self.brake_start_wp = self.generate_brake_path_with_constraint_distance(
                    self.current_vel, d)

        # traffic light previous is red
        elif msg.data == -1 and self.next_wp_idx > self.tl_waypoint and self.tl_waypoint!=-1:

            rospy.logwarn("traffic light at Waypoint:%d is GREEN", self.tl_waypoint)
            # stop braking
            self.brake_start_wp = self.total_wp_num
            # speed upa car
            self.speedup_wps, self.speedup_stop_wp = self.generate_speedup_path(self.current_vel,self.target_vel,
                                                                                self.current_pose.pose)
            self.tl_waypoint = -1

    def traffic_lights_cb(self,msg):
        # rospy.logwarn("traffic lights count %d", len(msg.lights))
        if self.waypoints is None or self.current_pose is None or self.next_tf_idx >= len(self.light_positions):
            return

        # first time to generate the path.
        if self.next_tf_idx == -1:
            self.next_tf_idx += 1
            next_tl_wp = self.light_pos_wps[self.next_tf_idx]
            self.check_tf_wp = self.checkwp_before_traffic_light(next_tl_wp)
            rospy.logwarn("Next traffic idx %d, waypoint at: %d, start to check  at:%d", self.next_tf_idx,
                          next_tl_wp, self.check_tf_wp)

        # state machine to handle behavior in front of traffic light
        if self.traffic_state == Traffic.IN_BRAKE_ZONE:
            if msg.lights[self.next_tf_idx].state < 2:
                next_tl_wp = self.light_pos_wps[self.next_tf_idx]
                d = WaypointUpdater.distance(self.waypoints, self.next_wp_idx, next_tl_wp)
                wps= self.generate_brake_path(next_tl_wp, d)
                if wps is None:
                    # self.augmented_wps= self.generate_speedup_path()
                    rospy.logwarn("traffic light at Waypoint:%d is YELLOW/RED", next_tl_wp)
                    rospy.logwarn("Not enough distance to brake, risk to speed up through traffic light")
                    self.traffic_state = Traffic.SPEED_UP
                    rospy.logwarn("state is in AFTER_TRAFFIC_LIGHT")
                else:
                    self.augmented_wps = wps
                    rospy.logwarn("traffic light at Waypoint:%d is RED", next_tl_wp)
                    rospy.logwarn("Begin to slow down, state is in IN_STOPPING")
                    self.traffic_state = Traffic.IN_STOPPING

            # check car travels through the traffic position
            if self.next_wp_idx >= self.light_pos_wps[self.next_tf_idx]:
                self.traffic_state = Traffic.SPEED_UP

        elif self.traffic_state == Traffic.IN_STOPPING:
            if msg.lights[self.next_tf_idx].state == 2:
                # self.augmented_wps = self.generate_speedup_path()
                self.traffic_state = Traffic.SPEED_UP
                rospy.logwarn("traffic light at Waypoint:%d is GREEN", self.light_pos_wps[self.next_tf_idx])
                rospy.logwarn("state is in SPEED_UP")

        elif self.traffic_state == Traffic.SPEED_UP:
            self.augmented_wps = None
            self.next_tf_idx += 1
            if self.next_tf_idx >= len(self.light_pos_wps):
                self.check_tf_wp = self.total_wp_num
            else:
                next_tl_wp = self.light_pos_wps[self.next_tf_idx]
                self.check_tf_wp = self.checkwp_before_traffic_light(next_tl_wp)
                rospy.logwarn("Next traffic idx %d, waypoint at: %d, start to check  at:%d", self.next_tf_idx,
                          next_tl_wp, self.check_tf_wp)
            self.traffic_state = Traffic.FREE
            rospy.logwarn("state is in FREE")

        else: # self.traffic_state == Traffic.FREE
            if self.next_wp_idx > self.check_tf_wp:
                self.traffic_state = Traffic.IN_BRAKE_ZONE
                rospy.logwarn("Begin to slow down, state is in IN_BRAKE_ZONE")
        return

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_vel_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    @staticmethod
    def interpolate_waypoints( waypoints, wp_idx,d0,distances,velocities,wp_is_start=True):
        '''
        :param waypoints:
        :param wp_idx:
        :param d0: distance of start or end waypoint of generated trajectory
        :param distances:
        :param velocities:
        :param wp_is_start:
        :return: intepolated waypoints with update of velocities,start_wp,stop_wp
        '''
        if len(distances) < 2:
            rospy.logwarn("Interpolating does not have enough points")
            return [], 0, 0

        ds_origs = []
        xs_origs = []
        ys_origs = []
        oz_origs = []
        d = d0
        wp_start = wp_idx
        wp_stop = wp_idx

        if wp_is_start:
            for i in range(wp_idx,len(waypoints)):
                ds_origs.append(d)
                xs_origs.append(waypoints[i].pose.pose.position.x)
                ys_origs.append(waypoints[i].pose.pose.position.y)
                oz_origs.append(waypoints[i].pose.pose.orientation.z)
                if d > distances[-1]:
                    wp_stop = i
                    break
                d += WaypointUpdater.distance(waypoints, i, i + 1)

        else:
            for i in range(wp_idx,-1,-1):
                ds_origs.append(d)
                xs_origs.append(waypoints[i].pose.pose.position.x)
                ys_origs.append(waypoints[i].pose.pose.position.y)
                oz_origs.append(waypoints[i].pose.pose.orientation.z)
                if d < distances[0]:
                    wp_start = i
                    break
                d -= WaypointUpdater.distance(waypoints, i-1, i)
            # reverse the list for spline interpolation
            ds_origs = ds_origs[::-1]
            xs_origs = xs_origs[::-1]
            ys_origs = ys_origs[::-1]
            oz_origs = oz_origs[::-1]

        if len(ds_origs) < 2:
            rospy.logwarn("Interpolating does not have enough waypoints")
            return [],0,0

        cs_x = CubicSpline(ds_origs,xs_origs,bc_type='natural')
        cs_y = CubicSpline(ds_origs,ys_origs,bc_type='natural')
        cs_oz = CubicSpline(ds_origs,oz_origs,bc_type='natural')

        xs_samples = cs_x(distances)
        ys_samples = cs_y(distances)
        oz_samples = cs_oz(distances)

        interpolated_wps=[]
        for i in range(len(distances)):
            p = Waypoint()
            p.pose.pose.position.x = xs_samples[i]
            p.pose.pose.position.y = ys_samples[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = oz_samples[i]
            p.twist.twist.linear.x = velocities[i]
            interpolated_wps.append(p)

        return interpolated_wps,wp_start,wp_stop

    @staticmethod
    def find_next_waypoint(waypoints,pose, start_wp=0):
        '''
        Get the next waypoint index
        :param pose: related position
        :param start_wp: start waypoint index for search, default 0
        :return: index of next waypoint
        '''

        d_min = float('inf')
        next_wp = start_wp

        for i in range(start_wp,len(waypoints)):
            # only for comparision not necessary to calulate sqaure root .
            d = WaypointUpdater.distance_2D_square(pose.pose.position, waypoints[i].pose.pose.position)
            next_wp = i

            if d < d_min:
                d_min = d
            else:
                # calculate angle between two vectors v1=x1 + y1*i, v2= x2 + y2*i
                x1 = waypoints[i].pose.pose.position.x - waypoints[i-1].pose.pose.position.x
                y1 = waypoints[i].pose.pose.position.y - waypoints[i-1].pose.pose.position.y
                x2 = pose.pose.position.x - waypoints[i-1].pose.pose.position.x
                y2 = pose.pose.position.y - waypoints[i-1].pose.pose.position.y
                # we only need to check if cos_theta sign to determin the angle is >90
                cos_theta_sign= x1*x2 + y1*y2

                if cos_theta_sign < 0:
                    next_wp = i -1
                # stop search, find the right one
                break

        # check if reaches the last wp.
        if 0< next_wp == len(waypoints)-1:
            x1 = waypoints[-2].pose.pose.position.x - waypoints[-1].pose.pose.position.x
            y1 = waypoints[-2].pose.pose.position.y - waypoints[-1].pose.pose.position.y
            x2 = pose.pose.position.x - waypoints[-1].pose.pose.position.x
            y2 = pose.pose.position.y - waypoints[-1].pose.pose.position.y
            # we only need to check if cos_theta sign to determin the angle is >90
            cos_theta_sign = x1 * x2 + y1 * y2
            if cos_theta_sign < 0:
                next_wp = next_wp +1

        return next_wp

    @staticmethod
    def predict_next_waypoint(waypoints,pose, vel, start_wp=0, delay = LATENCY):
        '''
        Get the next waypoint index
        :param pose: related position
        :param vel: current velocity
        :param start_wp: start waypoint index for search, default 0
        :return: predicted waypoint, next waypoint
        '''
        next_wp = WaypointUpdater.find_next_waypoint(waypoints,pose,start_wp)
        d0 = WaypointUpdater.direct_distance(pose.pose.position,waypoints[next_wp].pose.pose.position)
        predict_d = vel*delay
        predict_wp = next_wp

        if predict_d > d0:
            d = d0
            for i in range(next_wp+1,len(waypoints)):
                d += WaypointUpdater.distance(waypoints,i-1,i)
                predict_wp = i
                if d > predict_d:
                    break

        return predict_wp,next_wp

    @staticmethod
    def generate_dist_vels(v0, vt, accel, jerk):
        '''
        use a cubic spline to fit the distance vs speed
        :param v0: start velocity
        :param vt: target velocity
        :param accel: maximum acceleration
        :param jerk:  maximum jerk
        :return: list of calculated distances, list of calculated velocities
        '''
        ds,ts = WaypointUpdater.get_min_distance(v0,vt,accel,jerk,return_list=True)
        if len(ds) < 2:
            return [],[]

        cs_d = CubicSpline(ts, ds, bc_type='natural')
        cs_v = cs_d.derivative(nu=1)

        # generate sampled points from spline
        ts_samples = np.arange(T_STEP_SIZE, ts[-1]+T_STEP_SIZE, T_STEP_SIZE)
        ds_samples = cs_d(ts_samples)
        vs_samples = cs_v(ts_samples)

        if LOG:
            accel_s = cs_d(ts_samples, 2)
            js = cs_d(ts_samples, 3)
            rospy.loginfo("max accel %.03f, jerk %.03f",np.max(np.abs(accel_s)),np.max(np.abs(js)))

        return ds_samples, vs_samples

    @staticmethod
    def get_min_distance(v0, vt, accel, jerk, return_list = False):
        '''
        :param v0: start velocity
        :param vt: target velocity
        :param accel: max acceleration
        :param jerk: max jerk
        :return: list of key distances and list of time points
        '''

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
            return [], []

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
            # acceleration increases to max value, holds for t2 and decrease to 0
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
            return ds,ts
        else:
            return ds[-1]

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    @staticmethod
    def direct_distance(pos1,pos2):
        return  math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2)

    @staticmethod
    def distance_2D_square(pos1, pos2):
        return (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
