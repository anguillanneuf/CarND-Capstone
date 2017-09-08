from pid import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, kp,ki,kd,decel_limit,accel_limit,wheel_base, steer_ratio, min_speed, max_lat_accel,
                 max_steer_angle):
        # TODO: Implement
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.pid = PID(kp,ki,kd,-1,1)
        self.yaw = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.count = 0
        self.container = [0,0,0,0,0,0,0,0]
    def control(self, target_linear_vel,target_angl_vel,current_linear_vel,dbw_enabled,dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if dbw_enabled is False:
            self.pid.reset()
            return 0.0,0.0,0.0

        # for tuning purpose
        self.pid.kp = rospy.get_param('~kp', 0.2)
        self.pid.ki = rospy.get_param('~ki', 0.5)
        self.pid.kd = rospy.get_param('~kd', 0.001)


        error = target_linear_vel - current_linear_vel

        val = self.pid.step(error,dt)

        if target_linear_vel > 0:
            rospy.loginfo("Target v  %.03f  Error : %02d  Control:%.03f", target_linear_vel, int(100 * error / target_linear_vel),val)

        # reference for yaw control
        # Ackermann Steering dynamics http://correll.cs.colorado.edu/?p=1869
        steering = self.yaw.get_steering(target_linear_vel,target_angl_vel,current_linear_vel)
        if val >=0:
            return val*self.accel_limit,0.0,steering
        else:
            return 0.0,val*self.decel_limit,steering

