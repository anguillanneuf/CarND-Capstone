#!/usr/bin/env python

import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
T_STEP_SIZE = 0.02 # time step for slowdown or speedup


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32,self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.velocity = rospy.get_param('/waypoint_loader/velocity',20)*0.27778
        accelerate_rate = rospy.get_param('~accelerate_rate', 1.)
        brake_rate = rospy.get_param('~brake_rate', 1.)
        max_jerk = rospy.get_param('~max_jerk')

        self.accel_ds, self.accel_vs = self.get_smooth_cubic_spline(self.velocity,accelerate_rate,max_jerk)
        self.decel_ds, self.decel_vs = self.get_smooth_cubic_spline(self.velocity,brake_rate,max_jerk)

        self.waypoints = None
        self.final_waypoints = None
        self.total_wp_num = 0

        self.current_pose = None
        self.next_wp_idx = 0
        self.tl_waypoint= -1

        rospy.spin()

    def get_smooth_cubic_spline(self,speed,accel,jerk):
        '''
        use a cubic spline to fit the distance vs speed
        :param speed: target velocity
        :param accel: maximum acceleration
        :param jerk:  maximum jerk
        :return: a cubic spline object for fit later, and distance for spline
        '''

        t1 = accel/jerk
        t2 = (speed - jerk*t1**2)/accel+t1
        t3 = t2 + t1
        Ts = [0,t1,t2,t3]
        v1 = jerk*t1**2/2
        v2 = v1 + accel*(t2-t1)
        v3 = speed

        d1 = jerk * t1**3/6
        d2 = d1 + v1*(t2-t1)+accel*(t2-t1)**2/2
        d3 = d2 + v2*(t3-t2)+accel*(t3-t2)**2/2 - jerk *(t3-t2)**3/6

        Ds = [0,d1,d2,d3]

        cs_d = CubicSpline(Ts,Ds,bc_type='natural')
        cs_v = cs_d.derivative(nu=1)
        ts = np.arange(0,t3,T_STEP_SIZE)

        d_fitted = cs_d(ts)
        v_fitted = cs_v(ts)

        return d_fitted,v_fitted

    def update_speed_deceleration(self, pose, wp=None):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param pose: position where car stops
        :param wp: alternative to pose, a wp is index of related position
        :return: None
        '''

        d0 = 0 # distance to next wp
        if wp is not None:
            next_wp = wp
        else:
            next_wp = self.find_next_waypoint(pose)
            d0 = self.direct_distance(pose.position, self.waypoints[next_wp].pose.pose.position)

        # curve for de-acceleration
        brake_distance = self.decel_ds[-1]
        brake_start_wp = next_wp
        Xs = []
        Ys = []
        Ds = []
        Oz = []

        for i in range(1,next_wp):
            d = self.distance(self.waypoints, next_wp -i, next_wp) -d0
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

        Xfitted = Cs_x(self.decel_ds)
        Yfitted = Cs_y(self.decel_ds)
        Ozfitted = Cs_oz(self.decel_ds)

        augumented_waypoints=[]

        for i in range(len(self.decel_ds)):
            p = Waypoint()
            p.pose.pose.position.x = Xfitted[i]
            p.pose.pose.position.y = Yfitted[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = Ozfitted[i]
            p.twist.twist.linear.x = self.decel_vs[i]

            augumented_waypoints.append(p)

        # reverse the list
        return augumented_waypoints[::-1], brake_start_wp


    def update_speed_speedup(self, pose, wp=None):
        '''
        update speed if car needs to stop at a position, it follows a slow-down to position and speedup afterwards
        :param pose: position where car stops
        :param wp: alternative to pose, a wp is index of related position
        :return: None
        '''

        d0 = 0
        if wp is not None:
            next_wp = wp
        else:
            next_wp = self.find_next_waypoint(pose)
            d0 = self.direct_distance(pose.position, self.waypoints[next_wp].pose.pose.position)

        wp_trip_count = len(self.waypoints)

        # curve for acceleration
        accel_distance = self.accel_ds[-1]
        accel_stop_wp = next_wp
        Xs = []
        Ys = []
        Ds = []
        Oz = []

        for i in range(wp_trip_count- next_wp):
            d = self.distance(self.waypoints, next_wp, next_wp+i) +d0
            Ds.append(d)
            Xs.append(self.waypoints[next_wp + i].pose.pose.position.x)
            Ys.append(self.waypoints[next_wp + i].pose.pose.position.y)
            Oz.append(self.waypoints[next_wp + i].pose.pose.orientation.z)

            if d > accel_distance:
                accel_stop_wp = next_wp + i
                break

        Cs_x = CubicSpline(Ds, Xs, bc_type='natural')
        Cs_y = CubicSpline(Ds, Ys, bc_type='natural')
        Cs_oz = CubicSpline(Ds, Oz, bc_type='natural')

        Xfitted = Cs_x(self.accel_ds)
        Yfitted = Cs_y(self.accel_ds)
        Ozfitted = Cs_oz(self.accel_ds)

        augumented_waypoints = []

        for i in range(len(self.decel_ds)):
            p = Waypoint()
            p.pose.pose.position.x = Xfitted[i]
            p.pose.pose.position.y = Yfitted[i]
            p.pose.pose.position.z = 0
            p.pose.pose.orientation.z = Ozfitted[i]
            p.twist.twist.linear.x = self.accel_vs[i]

            augumented_waypoints.append(p)

        return augumented_waypoints, accel_stop_wp


    def pose_cb(self, msg):
        # TODO: Done Implement

        # rospy.loginfo('current_pose Received - x:%d, y:%d,z:%d', msg.pose.position.x, msg.pose.position.y,
        #              msg.pose.position.z)
        # get next waypoint index
        if self.final_waypoints is not None:
            # first time to update speed for acceleration to target speed and set the stop point
            if self.current_pose is None:
                self.current_pose = msg
                self.update_speed_speedup(self.current_pose.pose)
            else:
                self.current_pose = msg

            # find next waypoint index
            next_wp_idx = self.find_next_waypoint(self.waypoints,self.current_pose.pose,self.next_wp_idx)
            if next_wp_idx >self.next_wp_idx:
                rospy.logwarn("Next waypoint:%d",next_wp_idx)
            self.next_wp_idx = next_wp_idx

            # publish new final waypoints
            next_wp_idx = self.find_next_waypoint(self.final_waypoints,self.current_pose.pose,0)
            self.final_waypoints = self.final_waypoints[next_wp_idx:]

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now()

            stop  = min(LOOKAHEAD_WPS, len(self.final_waypoints))
            if self.tl_waypoint is not -1:
                stop = max(0,len(self.final_waypoints)- (self.total_wp_num -self.tl_waypoint))

            lane.waypoints = self.waypoints[:stop]
            self.final_waypoints_pub.publish(lane)



    def waypoints_cb(self, waypoints):
        # TODO: Done Implement
        # rospy.logwarn('waypoints Received - count:%d',len(waypoints.waypoints))
        if self.waypoints is None:
            self.waypoints = waypoints.waypoints
            self.final_waypoints = waypoints.waypoints
            self.total_wp_num = len(self.waypoints)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if self.tl_waypoint == -1:
            if msg.data >self.next_wp_idx:
                self.tl_waypoint = msg.data
                augumented_wps, brake_start_wp =\
                    self.update_speed_deceleration(self.waypoints[self.tl_waypoint], wp=self.tl_waypoint)

                remaining_wp = len(self.final_waypoints)
                self.final_waypoints = self.final_waypoints[:remaining_wp - (self.total_wp_num - brake_start_wp)+1] + \
                                       augumented_wps + self.final_waypoints[-(self.total_wp_num-self.tl_waypoint):]

        else:
            if msg.data == -1:
                # find next waypoint index

                augumented_wps,accel_stop_wp = self.update_speed_speedup(self.current_pose.pose)

                remaining_wp = len(self.final_waypoints)
                self.final_waypoints = self.final_waypoints[:remaining_wp - (self.total_wp_num - self.tl_waypoint) + 1] + \
                                       augumented_wps + self.final_waypoints[-(self.total_wp_num - self.accel_stop_wp):]
                self.tl_waypoint = -1


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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
        for i in range(0, len(waypoints), LOOKAHEAD_WPS):
            start = i+start_wp
            stop = min(start + LOOKAHEAD_WPS, start + len(waypoints))
            distances = [self.direct_distance(pose.position, waypoints[idx % wp_trip_count].pose.pose.position)
                         for idx in range(start, stop)]
            arg_min_idx = np.argmin(distances)
            # stop searching
            if arg_min_idx < (stop - start - 1):
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

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def direct_distance(self,pos1,pos2):
        return  math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
