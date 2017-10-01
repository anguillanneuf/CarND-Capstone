#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

LOG = False # Set to True to enable logs

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -10)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius',0.335)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        kp = rospy.get_param('~kp',1.2)
        ki = rospy.get_param('~ki',1.0)
        kd = rospy.get_param('~kd',0.01)
        min_speed = 0.1

        # x4 for 4 wheels of car
        #max_torque = vehicle_mass*decel_limit*wheel_radius *4 * 1.0

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `TwistController` object
        self.controller = Controller(kp,ki,kd, vehicle_mass, fuel_capacity, decel_limit, accel_limit, brake_deadband, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped,self.dbw_twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_vel_cb)

        self.dbw_enabled = False
        self.current_linear_vel = 0
        self.target_linear_vel = 0
        self.target_angle_vel = 0

        self.loop()

    def dbw_enabled_cb(self,msg):
        self.dbw_enabled = msg.data
        if LOG:
            rospy.loginfo('dbw_enabled recieved:%r', self.dbw_enabled)

    def current_vel_cb(self,msg):
        self.current_linear_vel = msg.twist.linear.x
        if LOG:
            rospy.loginfo('current_vel recieved:%f', self.current_linear_vel)

    def dbw_twist_cb(self,msg):
        # TODO:
        self.target_linear_vel = msg.twist.linear.x
        self.target_angle_vel = msg.twist.angular.z
        if LOG:
            rospy.loginfo('dbw_twist_cb recieved vel:%f, angl:%f', self.target_linear_vel, self.target_angle_vel)

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        dt = 0.02 # 20 ms
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            throttle, brake, steering = self.controller.control(self.target_linear_vel,
                                                                self.target_angle_vel,
                                                                self.current_linear_vel,
                                                                self.dbw_enabled,
                                                                dt)
            if self.dbw_enabled:
                self.publish(throttle, brake, steering)
                if LOG:
                    rospy.loginfo('publish throttle:%f,brake:%f,steering:%f', throttle, brake, steering)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
