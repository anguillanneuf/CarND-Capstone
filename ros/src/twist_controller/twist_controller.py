from pid_my import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, kp, ki, kd, vehicle_mass, fuel_capacity, accel_limit, brake_deadband, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # TODO: Implement
        # self.max_torque = abs(max_torque)
        self.accel_limit = accel_limit
        self.brake_deadband = brake_deadband
        self.pid = PID(kp,ki,kd, -10,0.5)

        self.yaw = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.wheel_radius = wheel_radius

    def control(self, target_linear_vel,target_angl_vel,current_linear_vel,dbw_enabled,dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if dbw_enabled is False:
            self.pid.reset()
            return 0.0,0.0,0.0

        error = target_linear_vel - current_linear_vel

        val = self.pid.step(error,dt)

        # reference for yaw control
        # Ackermann Steering dynamics http://correll.cs.colorado.edu/?p=1869
        steering = self.yaw.get_steering(target_linear_vel,target_angl_vel,current_linear_vel)
        if val >= 0:
            return val, 0.0, steering
        else:
            # rospy.logwarn("Target v  %.03f  Error : %.03f  Control:%.03f", target_linear_vel, error, val)
            if -val < self.brake_deadband:
                # rospy.logdebug("Dead band, %.03f",val*self.decel_limit)
                return 0.0, 0.0, steering
            return 0.0, self.calculate_torque(-val), steering

    def calculate_torque(self, acceleration):
        torque = self.total_mass * acceleration * self.wheel_radius
        return torque
