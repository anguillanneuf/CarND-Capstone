from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, ki,kp,kd,decel_limit,accel_limit,wheel_base, steer_ratio, min_speed, max_lat_accel,
                 max_steer_angle):
        # TODO: Implement
        self.pid = PID(kp,ki,kd,decel_limit,accel_limit)
        self.yaw = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, target_linear_vel,target_angl_vel,current_linear_vel,current_angl_vel,dbw_enabled,dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if dbw_enabled is False:
            self.pid.reset()
            return 0.0,0.0,0.0

        error = target_linear_vel - current_linear_vel
        val = self.pid.step(error,dt)
        steering = self.yaw.get_steering(current_linear_vel,target_angl_vel,current_angl_vel)
        if val >=0:
            return val,0.0,steering
        else:
            return 0.0,-val,steering

