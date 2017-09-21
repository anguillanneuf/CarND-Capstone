import rospy

MIN_NUM = float('-inf')
MAX_NUM = float('inf')

LOG = False # Set to True to enable logs

class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_int_val = self.last_error = 0.
        self.best_error = float('inf')

        self.twiddle_update_mode = 0
        self.update_position = 0
        self.update_threshold = 0.0001
        self.dkp = kp / 4.
        self.dki = ki / 4.
        self.dkd = 0.01
        self.error_list = []


    def reset(self):
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):
        self.last_int_val = self.int_val

        derivative = (error - self.last_error) / sample_time;

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = self.int_val + error * sample_time;
        self.last_error = error


        # use only latest 500 errors to update coeffs
        if len(self.error_list) > 500:
            self.error_list.pop(0)
        self.error_list.append(error*error)
        error = sum(self.error_list)/len(self.error_list)
        self.update_coeff(error)

        return val

    def update_coeff(self, error):
        """Update kp, ki, kd by Twiddle algorithm"""

        if ((self.dkp + self.dki + self.dkd) < self.update_threshold):
            return

        if self.twiddle_update_mode == 0:
            if self.update_position == 0: self.kp += self.dkp
            if self.update_position == 1: self.ki += self.dki
            if self.update_position == 2: self.kd += self.dkd
            self.twiddle_update_mode = 1
            return

        if self.twiddle_update_mode == 1:
            if error < self.best_error:
                self.best_error = error
                if self.update_position == 0: self.dkp *= 1.05
                if self.update_position == 1: self.dki *= 1.05
                if self.update_position == 2: self.dkd *= 1.05
                self.twiddle_update_mode = 0
                self.update_position += 1
                self.update_position = self.update_position % 3
            else:
                if self.update_position == 0: self.kp -= 2*self.dkp
                if self.update_position == 1: self.ki -= 2*self.dki
                if self.update_position == 2: self.kd -= 2*self.dkd
                self.twiddle_update_mode = 2
            return

        if self.twiddle_update_mode == 2:
            if error < self.best_error:
                self.best_error = error
                if self.update_position == 0: self.dkp *= 1.05
                if self.update_position == 1: self.dki *= 1.05
                if self.update_position == 2: self.dkd *= 1.05
            else:
                if self.update_position == 0:
                    self.kp += self.dkp
                    self.dkp *= 0.95
                elif self.update_position == 1:
                    self.ki += self.dki
                    self.dki *= 0.95
                elif self.update_position == 2:
                    self.kd += self.dkd
                    self.dkd *= 0.95
            self.twiddle_update_mode = 0
            self.update_position += 1
            self.update_position = self.update_position % 3
            return
