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
        self.velocity = rospy.get_param('/waypoint_loader/velocity',20)*0.27778
        self.accelerate_rate = rospy.get_param('~accelerate_rate', 1.)
        self.brake_rate = rospy.get_param('~brake_rate', 1.)
        self.max_jerk = rospy.get_param('~max_jerk')

        # self.accel_ds, self.accel_vs = self.get_smooth_cubic_spline(self.velocity,self.accelerate_rate,self.max_jerk)
        # self.decel_ds, self.decel_vs = self.get_smooth_cubic_spline(self.velocity,self.brake_rate,self.max_jerk)
        # calculate minimum brake distance
        distances,_ = self.get_smooth_cubic_spline(self.velocity,self.brake_rate,self.max_jerk)
        self.min_brake_distance = distances[-1]
        self.is_tl_red = False
        self.current_tf_wp = -1

        self.waypoints = None
        self.augmented_wps = None
        self.speedup_stop_wp = -1
        self.brake_start_wp = 0
        self.total_wp_num = 0

        self.current_pose = None
        self.next_wp_idx = 0
        self.tl_waypoint= -1
        self.current_vel = 0

        rospy.spin()

    def get_smooth_cubic_spline(self,speed,accel,jerk):
        '''
        use a cubic spline to fit the distance vs speed
        :param speed: target velocity
        :param accel: maximum acceleration
        :param jerk:  maximum jerk
        :return: list of calculated distances, list of calculated velocities
        '''

        # for speed is zero, return empty list
        if speed < 0.0001:
            return [], []

        t1 = accel/np.float(jerk)
        v1 = jerk * t1 ** 2 / 2
        t_end = t1

        if speed <= v1*2:
            #  acceleration will not reach max value
            t1 = math.sqrt(speed/jerk)
            v1 = jerk * t1 **2/2
            a1 = jerk * t1
            t_end = t1*2
            d1 = jerk * t1 ** 3 / 6
            d2 = d1 + v1 * (t_end - t1) + a1 * (t_end - t1) ** 2 / 2 - jerk * (t_end - t1) ** 3 / 6

            Ts = [0, t1, t_end]
            Ds = [0, d1, d2]

        else:
            # acceleration increases to max value, holds for t2 and decrease to 0
            t2 = (speed - jerk*t1**2)/accel+t1
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
        ts = np.arange(0,t_end,T_STEP_SIZE)

        d_fitted = cs_d(ts)
        v_fitted = cs_v(ts)

        return d_fitted,v_fitted

    def update_speed_deceleration(self, target_pose):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param pose: position where car stops
        :return: None
        '''


        next_wp = self.find_next_waypoint(self.waypoints,target_pose)
        d0 = WaypointUpdater.direct_distance(target_pose.position, self.waypoints[next_wp].pose.pose.position)

        d_fitted,v_fitted = self.get_smooth_cubic_spline(self.velocity,self.brake_rate,self.max_jerk)

        if len(d_fitted) < 2:
            return [], self.total_wp_num

        # curve for de-acceleration
        brake_distance = d_fitted[-1]
        brake_start_wp = next_wp

        Ds = []
        Xs = []
        Ys = []
        Oz = []

        for i in range(1,next_wp):
            d = WaypointUpdater.distance(self.waypoints, next_wp -i, next_wp) -d0
            Ds.append(d)
            Xs.append(self.waypoints[next_wp-i].pose.pose.position.x)
            Ys.append(self.waypoints[next_wp-i].pose.pose.position.y)
            Oz.append(self.waypoints[next_wp-i].pose.pose.orientation.z)

            if d > brake_distance:
                brake_start_wp = next_wp -i
                break

        Cs_x = CubicSpline(Ds,Xs,bc_type='natural')
        Cs_y = CubicSpline(Ds,Ys,bc_type='natural')
        Cs_oz = CubicSpline(Ds,Oz,bc_type='natural')

        Xfitted = Cs_x(d_fitted)
        Yfitted = Cs_y(d_fitted)
        Ozfitted = Cs_oz(d_fitted)

        augmented_waypoints=[]

        for i in range(len(d_fitted)):
            p = Waypoint()
            p.pose.pose.position.x = Xfitted[i]
            p.pose.pose.position.y = Yfitted[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = Ozfitted[i]
            p.twist.twist.linear.x = v_fitted[i]

            augmented_waypoints.append(p)

        # rospy.logdebug("fit points:")
        # for i in range(len(Ds)):
        #    rospy.logdebug("wp x:%.03f,y:%.03f,yaw:%.03f, distance:%.03f", Xs[i],
        #                  Ys[i], Oz[i], Ds[i])

        #rospy.logwarn("original waypoints:")
        #for wp in self.waypoints[brake_start_wp:next_wp]:
        #    rospy.logwarn("wp x:%.03f,y:%.03f,yaw:%.03f,v:%.03f", wp.pose.pose.position.x,
        #                  wp.pose.pose.position.y,wp.pose.pose.orientation.z, wp.twist.twist.linear.x)

        # rospy.logdebug("brake_start_wp: %d", brake_start_wp)
        rospy.logwarn("augmented waypoints,brake_start_wp: %d ",brake_start_wp)
        #for wp in augmented_waypoints[::-1]:
        #    rospy.logwarn("wp x:%.03f,y:%.03f,yaw:%.03f,v:%.03f", wp.pose.pose.position.x,
        #                  wp.pose.pose.position.y,wp.pose.pose.orientation.z, wp.twist.twist.linear.x)
        # reverse the list
        return augmented_waypoints[::-1], brake_start_wp


    def update_speed_speedup(self, pose):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param pose: position where car stops
        :return: None
        '''

        next_wp = self.find_next_waypoint(self.waypoints,pose)
        d0 = WaypointUpdater.direct_distance(pose.position, self.waypoints[next_wp].pose.pose.position)

        d_fitted, v_fitted = self.get_smooth_cubic_spline(self.velocity, self.accelerate_rate, self.max_jerk)

        if len(d_fitted) < 2:
            return [], -1
        # curve for acceleration

        # if current velocity is not zero, find the closest entry in generated trajectory where v ~ current velocity
        v_idx = np.argmin(np.abs(v_fitted - self.current_vel))
        speedup_distance = d_fitted[-1]
        speedup_stop_wp = next_wp

        Ds = []
        Xs = []
        Ys = []
        Oz = []
        if d0 >0:
            Ds.append(d_fitted[0])
            Xs.append(pose.position.x)
            Ys.append(pose.position.y)
            Oz.append(pose.orientation.z)

        for i in range(0, next_wp):
            d = WaypointUpdater.distance(self.waypoints, next_wp, next_wp + i) + d0 + d_fitted[v_idx]

            Ds.append(d)
            Xs.append(self.waypoints[next_wp + i].pose.pose.position.x)
            Ys.append(self.waypoints[next_wp + i].pose.pose.position.y)
            Oz.append(self.waypoints[next_wp + i].pose.pose.orientation.z)
            if d > speedup_distance:
                speedup_stop_wp = next_wp + i
                break

        Cs_x = CubicSpline(Ds, Xs, bc_type='natural')
        Cs_y = CubicSpline(Ds, Ys, bc_type='natural')
        Cs_oz = CubicSpline(Ds, Oz, bc_type='natural')

        # remove points which velocity < current_vel
        d_fitted = d_fitted[v_idx:]
        v_fitted = v_fitted[v_idx:]

        Xfitted = Cs_x(d_fitted)
        Yfitted = Cs_y(d_fitted)
        Ozfitted = Cs_oz(d_fitted)

        augmented_waypoints = []

        for i in range(1,len(d_fitted)):
            p = Waypoint()
            p.pose.pose.position.x = Xfitted[i]
            p.pose.pose.position.y = Yfitted[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = Ozfitted[i]
            p.twist.twist.linear.x = v_fitted[i]

            augmented_waypoints.append(p)

        # rospy.logdebug("fit points:")
        # for i in range(len(Ds)):
        #     rospy.logdebug("wp x:%.03f,y:%.03f,yaw:%.03f, distance:%.03f", Xs[i],
        #                 Ys[i], Oz[i],Ds[i])
        #
        # rospy.logwarn("original waypoints:")
        # for wp in self.waypoints[next_wp:speedup_stop_wp]:
        #     rospy.logdebug("wp x:%.03f,y:%.03f,z:%.03f,yaw:%.03f,v:%.03f", wp.pose.pose.position.x,
        #                   wp.pose.pose.position.y, wp.pose.pose.position.z,
        #                   wp.pose.pose.orientation.z, wp.twist.twist.linear.x)
        #
        # rospy.logdebug("speedup_stop_wp: %d", speedup_stop_wp)
        rospy.logwarn("augmented waypoints, speedup_stop_wp %d",speedup_stop_wp)
        # for wp in augmented_waypoints:
        #     rospy.logwarn("wp x:%.03f,y:%.03f,yaw:%.03f,v:%.03f", wp.pose.pose.position.x,
        #                   wp.pose.pose.position.y, wp.pose.pose.orientation.z, wp.twist.twist.linear.x)


        return augmented_waypoints, speedup_stop_wp


    def pose_cb(self, msg):
        # TODO: Done Implement

        # rospy.loginfo('current_pose Received - x:%d, y:%d,z:%d', msg.pose.position.x, msg.pose.position.y,
        #              msg.pose.position.z)
        # get next waypoint index
        if self.waypoints is not None:
            # first time to update speed for acceleration to target speed and set the stop point
            if self.current_pose is None:
                self.current_pose = msg
                self.augmented_wps,self.speedup_stop_wp = self.update_speed_speedup(self.current_pose.pose)
                rospy.logwarn("Speed up wp count,%d",len(self.augmented_wps))
            else:
                self.current_pose = msg

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now()
            # find next waypoint index
            next_wp_idx = self.find_next_waypoint(self.waypoints,self.current_pose.pose,self.next_wp_idx)

            if next_wp_idx > self.next_wp_idx:
                rospy.logwarn("Np:%d, x:%.03f,y:%.03f", next_wp_idx,self.current_pose.pose.position.x,self.current_pose.pose.position.y)

            # check if traffic light is present:
            if next_wp_idx >= self.brake_start_wp:
                # publish decelleration waypoints
                passed_wp_idx = self.find_next_waypoint(self.augmented_wps,self.current_pose.pose,0)
                self.augmented_wps = self.augmented_wps[passed_wp_idx:]
                lane.waypoints = self.augmented_wps
                # if len(lane.waypoints) > 0:
                #    rospy.logwarn("publish brake wps %d",len(lane.waypoints))

            elif next_wp_idx < self.speedup_stop_wp:
                # publish acceleration waypoints:
                passed_wp_idx = self.find_next_waypoint(self.augmented_wps, self.current_pose.pose, 0)
                self.augmented_wps = self.augmented_wps[passed_wp_idx:]
                count = len(self.augmented_wps)
                lane.waypoints = self.augmented_wps
                #if passed_wp_idx>0:
                #    rospy.logwarn("publish speedup wps, %d", count)
                if count< LOOKAHEAD_WPS:
                    stop = min(self.speedup_stop_wp + LOOKAHEAD_WPS -count,self.total_wp_num)
                    lane.waypoints = lane.waypoints + self.waypoints[self.speedup_stop_wp:stop]

            else:
                # publish normal waypoints
                stop = min(next_wp_idx + LOOKAHEAD_WPS,self.total_wp_num)
                lane.waypoints = self.waypoints[next_wp_idx:stop]

                # rospy.logwarn("publish normal wps, %d,",len(lane.waypoints))

            self.next_wp_idx = next_wp_idx
            self.final_waypoints_pub.publish(lane)


    def waypoints_cb(self, waypoints):
        # TODO: Done Implement
        # rospy.logwarn('waypoints Received - count:%d',len(waypoints.waypoints))
        if self.waypoints is None:
            self.waypoints = waypoints.waypoints
            self.total_wp_num = len(self.waypoints)
            self.brake_start_wp = self.total_wp_num



    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # traffic light state = red is detected
        if self.tl_waypoint == -1 and msg.data > self.next_wp_idx:
            self.tl_waypoint = msg.data
            self.augmented_wps, self.brake_start_wp =\
                    self.update_speed_deceleration(self.waypoints[self.tl_waypoint].pose.pose)

        # traffic light previous is red
        elif self.tl_waypoint is not -1:
            self.tl_waypoint = msg.data

            # traffic light is green
            if msg.data == -1:
                self.brake_start_wp = self.total_wp_num
                self.augmented_wps,self.speedup_stop_wp = self.update_speed_speedup(self.current_pose.pose)
            # traffic light is updated with new position
            else:
                self.augmented_wps, self.brake_start_wp = \
                self.update_speed_deceleration(self.waypoints[self.tl_waypoint].pose.pose)

    def traffic_lights_cb(self,msg):
        # rospy.logwarn("traffic lights count %d", len(msg.lights))
        if self.waypoints is None or self.current_pose is None:
            return

        next_tl_wp = self.current_tf_wp
        next_light = None

        for light in msg.lights:
            light.pose.pose.position.z = 0
            next_tl_wp = self.find_next_waypoint(self.waypoints, light.pose.pose)
            if next_tl_wp > self.next_wp_idx:
                next_light = light
                break

        if self.is_tl_red:

            if next_tl_wp > self.current_tf_wp:
                rospy.logwarn("travel through last trafflic light at Waypoint:%d",self.current_tf_wp)
                self.is_tl_red = False
                self.brake_start_wp = self.total_wp_num
                self.augmented_wps, self.speedup_stop_wp = self.update_speed_speedup(self.current_pose.pose)
            elif next_light.state == 2:
                self.is_tl_red = False
                rospy.logwarn("traffic light at Waypoint:%d is Green", next_tl_wp)
                self.brake_start_wp = self.total_wp_num
                self.augmented_wps, self.speedup_stop_wp = self.update_speed_speedup(self.current_pose.pose)

        elif next_light is not None:
            d = self.distance(self.waypoints, self.next_wp_idx, next_tl_wp)
            if d < self.min_brake_distance + 40 and next_light.state < 2:
                self.is_tl_red = True
                rospy.logwarn("traffic light at Waypoint:%d is Red", next_tl_wp)
                self.augmented_wps, self.brake_start_wp = self.update_speed_deceleration(
                        self.waypoints[next_tl_wp-25].pose.pose)

        self.current_tf_wp = next_tl_wp


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_vel_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def find_next_waypoint(self, waypoints,pose, start_wp=0):
        '''
        Get the next waypoint index
        :param pose: related position
        :param start_wp: start waypoint index for search, default 0
        :return: index of next waypoint
        '''
        (_, _, yaw) = tf.transformations.euler_from_quaternion([pose.orientation.x,
                                                                pose.orientation.y,
                                                                pose.orientation.z,
                                                                pose.orientation.w])

        wp_trip_count = len(waypoints)
        closest_wp_idx = start_wp
        for i in range(0, len(waypoints), LOOKAHEAD_WPS):
            start = i+start_wp
            stop = min(start + LOOKAHEAD_WPS, start + len(waypoints))
            distances = [WaypointUpdater.direct_distance(pose.position, waypoints[idx % wp_trip_count].pose.pose.position)
                         for idx in range(start, stop)]
            arg_min_idx = np.argmin(distances)
            # stop searching
            if arg_min_idx < (stop - start - 1) or stop == len(waypoints):
                closest_wp_idx = (start + arg_min_idx) % wp_trip_count
                y_wp = waypoints[closest_wp_idx].pose.pose.position.y
                x_wp = waypoints[closest_wp_idx].pose.pose.position.x
                y = pose.position.y
                x = pose.position.x
                heading = math.atan2(y_wp - y, x_wp - x)

                if math.fabs(heading - yaw) > math.pi / 3:
                    closest_wp_idx += 1
                break


        return closest_wp_idx

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
