#!/usr/bin/env python

import rospy
import numpy as np
import tf
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint,TrafficLightArray
from geometry_msgs.msg import TwistStamped
from python_common.helper import MathHelper

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
T_STEP_SIZE = 0.05 #time step for slowdown or speedup
LOG = False # Set to true to enable logs

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
        self.brake_rate = rospy.get_param('~brake_rate', 1.)
        self.max_jerk = rospy.get_param('~max_jerk')
        # only for test in simulator
        self.light_positions = rospy.get_param('~light_positions')
        self.light_pos_wps =[]

        self.waiting_for_tf = False
        self.next_tf_idx = -1

        self.waypoints = None
        self.speedup_wps = None
        self.brake_wps = None

        self.speedup_stop_wp = -1
        self.brake_start_wp = 0
        self.total_wp_num = 0

        self.current_pose = None
        self.next_wp_idx = 0
        self.tl_waypoint= -1
        self.current_vel = 0

        rospy.spin()

    def get_distance_velocity_trajectory(self,v0,vt,accel,jerk):
        '''
        use a cubic spline to fit the distance vs speed
        :param v0: start velocity
        :param vt: target velocity
        :param accel: maximum acceleration
        :param jerk:  maximum jerk
        :return: list of calculated distances, list of calculated velocities
        '''

        # for speed is zero, return empty list
        if vt < 0.0001:
            return [], []

        t1 = accel/np.float(jerk)
        v1 = jerk * t1 ** 2 / 2
        t_end = t1

        if vt <= v1*2:
            #  acceleration will not reach max value
            t1 = math.sqrt(vt/jerk)
            v1 = jerk * t1 **2/2
            a1 = jerk * t1
            t_end = t1*2
            d1 = jerk * t1 ** 3 / 6
            d2 = d1 + v1 * (t_end - t1) + a1 * (t_end - t1) ** 2 / 2 - jerk * (t_end - t1) ** 3 / 6

            Ts = [0, t1, t_end]
            Ds = [0, d1, d2]

        else:
            # acceleration increases to max value, holds for t2 and decrease to 0
            t2 = (vt - jerk*t1**2)/accel+t1
            v2 = v1 + accel * (t2 - t1)
            t_end = t2 + t1
            # v3 = speed

            d1 = jerk * t1 ** 3 / 6
            d2 = d1 + v1 * (t2 - t1) + accel * (t2 - t1) ** 2 / 2
            d3 = d2 + v2 * (t_end - t2) + accel * (t_end - t2) ** 2 / 2 - jerk * (t_end - t2) ** 3 / 6

            Ts = [0,t1,t2,t_end]
            Ds = [0, d1, d2, d3]

        cs_d = CubicSpline(Ts,Ds,bc_type='natural')
        cs_v = cs_d.derivative(nu=1)

        # generate sampled points from spline
        Ts_sampled = np.arange(0,t_end,T_STEP_SIZE)
        Ds_sampled = cs_d(Ts_sampled)
        Vs_sampled = cs_v(Ts_sampled)

        # if current velocity is not zero, find the closest entry in generated trajectory where v ~ current velocity
        idx = np.argmin(np.abs(Vs_sampled - v0))

        return Ds_sampled[idx:],Vs_sampled[idx:]

    def get_min_distance_for_smooth_tractory(self, v0, vt, accel, jerk):
        '''
        :param v0: start velocity
        :param vt: target velocity
        :param accel: max acceleration
        :param jerk: max jerk
        :param slowdown: true to generate a slowdown trajectory
        :return: minimum distance for speedup or slowdown
        '''
        tj = accel / jerk
        dv = vt - v0

        if dv < 0:
            rospy.logwarn("current speed:%.03f m/s is over the target speed: %.03f m/s.",v0,vt)
            return 0
        # speed up does not reach maximum acceleration
        if dv < jerk * tj ** 2:
            t1 = math.sqrt(dv / jerk)
            t2 = t1
            d1 = v0 * t1 + jerk * t1 ** 3 / 6
            v1 = v0 + jerk * t1 ** 2 / 2
            a1 = jerk * t1
            d2 = v1 * t2 + a1 * t2 ** 2 / 2 - jerk * t2 ** 3 / 6

            return d1+d2
        else:
            t1 = tj
            t2 = (dv - jerk * tj ** 2) / accel
            t3 = tj
            v1 = v0 + jerk * t1 ** 2 / 2
            a1 = accel
            v2 = v1 + a1 * t2
            a2 = accel
            d1 = jerk * t1 ** 3 / 6
            d2 = v1 * t2 + a2 * t2 ** 2 / 2
            d3 = v2 * t3 + a2 * t3 ** 2 / 2 - jerk * t3 ** 3 / 6

            return d1+d2+d3


    def generate_brake_path_with_constraint_distance(self, v0, stop_wp,distance):
        '''

        :param v0: start velocity
        :param vt: target velocity
        :param distance: distance required
        :return: list of distances, list of velocity
        '''
        vt = self.target_vel
        d_speedup = self.get_min_distance_for_smooth_tractory(v0,vt, self.accelerate_rate, self.max_jerk)
        d_slowdown= self.get_min_distance_for_smooth_tractory(0,vt, self.brake_rate, self.max_jerk)

        delta_d = distance - d_speedup - d_slowdown
        # car has long enough distance to speed up first then slow down
        if delta_d > 0:
            return self.generate_brake_path(vt,stop_wp)
        else:
            # no speedup, just slow down
            d_slowdown = self.get_min_distance_for_smooth_tractory(0, v0, self.brake_rate, self.max_jerk)
            if distance < d_slowdown:
                # no way to avoid underbrake, but brake anyway
                return self.generate_brake_path(v0,stop_wp)
            else:
                # we can speedup shortly and then slowdown,
                # what is the proper velocity to speed up? we sample from max target velocity to current velocity
                dv = vt - v0
                scale = 10

                for i in range(1,scale):
                    vt = vt - dv/scale
                    d_speedup = self.get_min_distance_for_smooth_tractory(v0, vt, self.accelerate_rate, self.max_jerk)
                    d_slowdown = self.get_min_distance_for_smooth_tractory(0, vt, self.brake_rate, self.max_jerk)

                    delta_d = distance - d_speedup - d_slowdown
                    if delta_d >0:
                        break
                rospy.loginfo("Minimum speed up and slow down find proper v0:%.03f,vt: %.03f", v0,vt)
                speedup_wps,_ = self.generate_speedup_path(v0,vt,self.current_pose.pose, distance-d_slowdown)
                brake_wps,_ = self.generate_brake_path(vt,stop_wp)

                return speedup_wps+brake_wps,self.next_wp_idx-1


    def generate_brake_path(self, v0,stop_wp):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param v0: speed it starts to brake.
        :param stop_wp: waypoint index of position where car stops
        :return: None
        '''

        Ds_sampled,Vs_sampled = self.get_distance_velocity_trajectory(0,v0,self.brake_rate,self.max_jerk)

        if len(Ds_sampled) < 2:
            return [], self.total_wp_num

        Ds_original = []
        Xs_original = []
        Ys_original = []
        Oz_original = []

        d = 0
        Ds_original.append(d)
        Xs_original.append(self.waypoints[stop_wp].pose.pose.position.x)
        Ys_original.append(self.waypoints[stop_wp].pose.pose.position.y)
        Oz_original.append(self.waypoints[stop_wp].pose.pose.orientation.z)
        brake_start_wp = stop_wp

        for i in range(stop_wp-1,0,-1):
            d += WaypointUpdater.distance(self.waypoints, i, i+1)
            Ds_original.append(d)
            Xs_original.append(self.waypoints[i].pose.pose.position.x)
            Ys_original.append(self.waypoints[i].pose.pose.position.y)
            Oz_original.append(self.waypoints[i].pose.pose.orientation.z)

            if d > Ds_sampled[-1]:
                brake_start_wp = i
                break

        if len(Ds_original)<2:
            rospy.loginfo("No enough waypoints to fit the brake path, initial velocity, %.03f",v0)
            return [], -1


        Cs_x = CubicSpline(Ds_original,Xs_original,bc_type='natural')
        Cs_y = CubicSpline(Ds_original,Ys_original,bc_type='natural')
        Cs_oz = CubicSpline(Ds_original,Oz_original,bc_type='natural')

        Xs_sampled = Cs_x(Ds_sampled)
        Ys_sampled = Cs_y(Ds_sampled)
        Oz_sampled = Cs_oz(Ds_sampled)

        augmented_waypoints=[]

        for i in range(len(Ds_sampled)):
            p = Waypoint()
            p.pose.pose.position.x = Xs_sampled[i]
            p.pose.pose.position.y = Ys_sampled[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = Oz_sampled[i]
            p.twist.twist.linear.x = np.fabs(Vs_sampled[i])

            augmented_waypoints.append(p)

        augmented_waypoints= augmented_waypoints[::-1]

        # add additional waypoints and set velocitiy=0, this is used when brake waypoints are published,
        # but car still not stops.
        additional_wps = self.waypoints[stop_wp+1:stop_wp+50]
        for i in range(len(additional_wps)):
            p = Waypoint()
            p.pose.pose.position.x = additional_wps[i].pose.pose.position.x
            p.pose.pose.position.y = additional_wps[i].pose.pose.position.y
            p.pose.pose.position.z = additional_wps[i].pose.pose.position.z
            p.pose.pose.orientation.z = additional_wps[i].pose.pose.orientation.z
            p.twist.twist.linear.x = 0

            augmented_waypoints.append(p)

        return augmented_waypoints, brake_start_wp

    def generate_speedup_path(self, v0,vt, current_position, distance = None):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param v0:start velocity
        :param vt:target velocity
        :param current_position
        :param distance if specified, the path has distance length
        :return: None
        '''

        next_wp = WaypointUpdater.find_next_waypoint(self.waypoints,current_position,self.next_wp_idx)
        d0 = WaypointUpdater.direct_distance(current_position.position, self.waypoints[next_wp].pose.pose.position)

        Ds_sampled, Vs_sampled = self.get_distance_velocity_trajectory(v0, vt, self.accelerate_rate, self.max_jerk)

        if len(Ds_sampled) < 2:
            return [], -1

        speedup_end_wp = next_wp

        # find the path and fit the x,y,yaw along the path
        Ds_original = []
        Xs_original = []
        Ys_original = []
        Oz_orignal = []
        if d0 >0:
            Ds_original.append(Ds_sampled[0])
            Xs_original.append(current_position.position.x)
            Ys_original.append(current_position.position.y)
            Oz_orignal.append(current_position.orientation.z)

        for i in range(next_wp,len(self.waypoints)):
            d = WaypointUpdater.distance(self.waypoints, next_wp, i) + d0 +Ds_sampled[0]

            Ds_original.append(d)
            Xs_original.append(self.waypoints[i].pose.pose.position.x)
            Ys_original.append(self.waypoints[i].pose.pose.position.y)
            Oz_orignal.append(self.waypoints[i].pose.pose.orientation.z)
            if d > Ds_sampled[-1]:
                speedup_end_wp = i
                break

        if len(Ds_original)<2:
            rospy.loginfo("No enough waypoints to fit the speedup path, initial v, %.03f, target v %.03f", v0,vt)
            return [], -1

        Cs_x = CubicSpline(Ds_original, Xs_original, bc_type='natural')
        Cs_y = CubicSpline(Ds_original, Ys_original, bc_type='natural')
        Cs_oz = CubicSpline(Ds_original, Oz_orignal, bc_type='natural')

        X_sampled = Cs_x(Ds_sampled)
        Y_sampled = Cs_y(Ds_sampled)
        OZ_sampled = Cs_oz(Ds_sampled)

        augmented_waypoints = []

        # start from 1 is because 0 is current position
        for i in range(1,len(Ds_sampled)):
            p = Waypoint()
            p.pose.pose.position.x = X_sampled[i]
            p.pose.pose.position.y = Y_sampled[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = OZ_sampled[i]
            p.twist.twist.linear.x = Vs_sampled[i]

            augmented_waypoints.append(p)

        if distance is not None and distance > Ds_sampled[-1]:
            v = Vs_sampled[-1]
            t = (distance- Ds_sampled[-1])/v
            num = t//T_STEP_SIZE +1
            Ds_sampled_ex = np.linspace(Ds_sampled[-1],distance,num=num,endpoint=True)
            rospy.logwarn("Additional %d point",num)

            X_sampled = Cs_x(Ds_sampled_ex)
            Y_sampled = Cs_y(Ds_sampled_ex)
            OZ_sampled = Cs_oz(Ds_sampled_ex)
            for i in range(1,len(Ds_sampled_ex)):
                p = Waypoint()
                p.pose.pose.position.x = X_sampled[i]
                p.pose.pose.position.y = Y_sampled[i]
                p.pose.pose.position.z = 0
                p.pose.pose.orientation.z = OZ_sampled[i]
                p.twist.twist.linear.x = Vs_sampled[-1]
                augmented_waypoints.append(p)

        rospy.logwarn("augmented waypoints, speedup_end_wp %d", speedup_end_wp)

        return augmented_waypoints, speedup_end_wp


    def pose_cb(self, msg):
        # TODO: Done Implement

        # rospy.loginfo('current_pose Received - x:%d, y:%d,z:%d', msg.pose.position.x, msg.pose.position.y,
        #              msg.pose.position.z)
        if self.waypoints is None:
            return

        # get next waypoint index
        # first time to update speed for acceleration to target speed and set the stop point
        if self.current_pose is None:
            self.current_pose = msg
            self.speedup_wps, self.speedup_stop_wp = self.generate_speedup_path(self.current_vel, self.target_vel,
                                                                                self.current_pose.pose)
            rospy.logwarn("Speed up to wp:%d", self.speedup_stop_wp )
        else:
            self.current_pose = msg

        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        # find next waypoint index
        next_wp_idx = WaypointUpdater.find_next_waypoint(self.waypoints, self.current_pose.pose, self.next_wp_idx)

        if next_wp_idx > self.next_wp_idx:
            rospy.logwarn("Next WayPoint:%d", next_wp_idx)

        # check if traffic light is present:
        if next_wp_idx >= self.brake_start_wp:
            # publish slow down waypoints
            passed_wp_idx = WaypointUpdater.find_next_waypoint(self.brake_wps, self.current_pose.pose, 0)
            self.brake_wps = self.brake_wps[passed_wp_idx:]
            lane.waypoints = self.brake_wps
            rospy.logwarn("publish brake wps %d", len(lane.waypoints))

        elif next_wp_idx < self.speedup_stop_wp:
            # publish acceleration waypoints:
            passed_wp_idx = WaypointUpdater.find_next_waypoint(self.speedup_wps, self.current_pose.pose, 0)
            self.speedup_wps = self.speedup_wps[passed_wp_idx:]
            count = len(self.speedup_wps)
            lane.waypoints = self.speedup_wps
            if passed_wp_idx > 0:
                rospy.logwarn("publish speedup wps, %d", count)
            if count < LOOKAHEAD_WPS:
                stop = min(self.speedup_stop_wp + LOOKAHEAD_WPS - count, self.total_wp_num)
                lane.waypoints = lane.waypoints + self.waypoints[self.speedup_stop_wp:stop]

        else:
            # publish normal waypoints
            start = min(len(self.waypoints) - 1, next_wp_idx)
            stop = min(next_wp_idx + LOOKAHEAD_WPS, self.total_wp_num)
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
            # following code is used for test traffic lights
            next_wp = 0
            for tf in self.light_positions:
                p = PoseStamped()
                p.pose.position.x = tf[0]
                p.pose.position.y = tf[1]
                next_wp = WaypointUpdater.find_next_waypoint(self.waypoints,p.pose,next_wp)
                self.light_pos_wps.append(next_wp)

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
        if self.waypoints is None or self.current_pose is None:
            return

        if self.next_tf_idx >= len(self.light_positions):
            return

        # first time to generate the path.
        if self.next_tf_idx == -1:
            self.next_tf_idx += 1
            rospy.logwarn("next traffic idx %d, stop at: %d", self.next_tf_idx,self.light_pos_wps[self.next_tf_idx])
            d = self.distance(self.waypoints, self.next_wp_idx, self.light_pos_wps[self.next_tf_idx])
            self.brake_wps, self.brake_start_wp = self.generate_brake_path_with_constraint_distance(self.current_vel,
                                             self.light_pos_wps[self.next_tf_idx], d)

        if self.waiting_for_tf:
            if msg.lights[self.next_tf_idx].state ==2:
                rospy.logwarn("traffic light at Waypoint:%d is GREEN", self.light_pos_wps[self.next_tf_idx])
                self.waiting_for_tf = False
                self.brake_start_wp = self.total_wp_num
                self.brake_wps = None
                self.speedup_wps, self.speedup_stop_wp = self.generate_speedup_path(self.current_vel,self.target_vel,
                                                                                    self.current_pose.pose)
                self.next_tf_idx +=1
                rospy.logwarn("next traffic idx %d, stop at: %d", self.next_tf_idx,
                              self.light_pos_wps[self.next_tf_idx])
                d = WaypointUpdater.distance(self.waypoints, self.next_wp_idx, self.light_pos_wps[self.next_tf_idx])
                self.brake_wps, self.brake_start_wp = self.generate_brake_path_with_constraint_distance(
                    self.current_vel,self.light_pos_wps[self.next_tf_idx],d)
        else:
            if self.next_wp_idx>=self.brake_start_wp:
                # traffic light is yellow or red
                if msg.lights[self.next_tf_idx].state < 2:
                    rospy.logwarn("traffic light at Waypoint:%d is RED", self.light_pos_wps[self.next_tf_idx])
                    self.waiting_for_tf = True

                else:
                    # update brake distance if car does not drive in top speed
                    brake_length = self.get_min_distance_for_smooth_tractory(0, self.current_vel, self.brake_rate,self.max_jerk)
                    d = self.distance(self.waypoints, self.next_wp_idx, self.light_pos_wps[self.next_tf_idx])
                    if d > brake_length:
                        self.brake_wps,self.brake_start_wp = self.generate_brake_path(self.current_vel,self.light_pos_wps[self.next_tf_idx])
                    # go quickly through the traffic light
                    else:
                        self.next_tf_idx += 1
                        rospy.logwarn("next traffic idx %d, stop at: %d", self.next_tf_idx,self.light_pos_wps[self.next_tf_idx])
                        d = WaypointUpdater.distance(self.waypoints, self.next_wp_idx, self.light_pos_wps[self.next_tf_idx])
                        self.brake_wps, self.brake_start_wp = self.generate_brake_path_with_constraint_distance(
                            self.current_vel, self.light_pos_wps[self.next_tf_idx], d)



    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_vel_cb(self, msg):
        self.current_vel = msg.twist.linear.x

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
            d = WaypointUpdater.distance_2D_square(pose.position, waypoints[i].pose.pose.position)
            next_wp = i

            if d < d_min:
                d_min = d
            else:
                # calculate angle between two vectors v1=x1 + y1*i, v2= x2 + y2*i
                x1 = waypoints[i].pose.pose.position.x - waypoints[i-1].pose.pose.position.x
                y1 = waypoints[i].pose.pose.position.y - waypoints[i-1].pose.pose.position.y
                x2 = pose.position.x - waypoints[i-1].pose.pose.position.x
                y2 = pose.position.y - waypoints[i-1].pose.pose.position.y
                # we only need to check if cos_theta sign to determin the angle is >90
                cos_theta_sign= x1*x2 + y1*y2

                if cos_theta_sign < 0:
                    next_wp = i -1
                # stop search, find the right one
                break

        # check if reaches the last wp.
        if next_wp == len(waypoints)-1:
            x1 = waypoints[-2].pose.pose.position.x - waypoints[-1].pose.pose.position.x
            y1 = waypoints[-2].pose.pose.position.y - waypoints[-1].pose.pose.position.y
            x2 = pose.position.x - waypoints[-1].pose.pose.position.x
            y2 = pose.position.y - waypoints[-1].pose.pose.position.y
            # we only need to check if cos_theta sign to determin the angle is >90
            cos_theta_sign = x1 * x2 + y1 * y2
            if cos_theta_sign < 0:
                next_wp = next_wp +1

        return next_wp

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
