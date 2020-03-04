import time

import numpy as np
from pybear import Manager
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

from pyrol.envs.real.real_env import RealEnv
from pyrol.envs.real.bear_pendulum import constants

# TODO: Where to put velocity limit breach?

class BearPendulumEnv(RealEnv):
    def __init__(self, 
                 port='/dev/ttyUSB0',
                 motor_id=1,
                 theta0_range=(-0.1,0.1),
                 max_speed=100.0,
                 max_torque=0.8,
                 dt=0.05,
                 exit_reward=50.0,
                 pos_exit_threshold=0.1,
                 vel_exit_threshold=0.1,
                 use_lqr_discrete=False,
                 l=0.37,
                 m=0.4,
                 b=0.07,
                 g = 9.8907,
                 ):

        self.modes = {'torque'       : 0,
                      'position'     : 1,
                      'velocity'     : 2,
                      'direct_force' : 3}

        # Pendulum parameters
        self.l = l
        self.m = m
        self.b = b  # 0.07 critically damped
        self.g = g

        self.port = port
        self.motor_id = motor_id
        self.theta0_range = theta0_range
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.exit_reward = exit_reward
        self.pos_exit_threshold = pos_exit_threshold
        self.vel_exit_threshold = vel_exit_threshold
        self.homing_torque = -0.5*self.m*self.g*self.l
        self.time_to_stabilize = 2.0
        self.encoder_offset = 112442
        self.use_lqr_discrete = use_lqr_discrete

        # self.Q = np.matrix([[1.0, 0.], [0., 0.1]], dtype=np.float64)
        # self.R = np.matrix([[0.001]], dtype=np.float64)

        self.QC = np.matrix([[1.0, 0.], [0., 0.6]], dtype=np.float64)
        self.RC = np.matrix([[.08]], dtype=np.float64)

        self.state_dim = (2,)
        self.action_dim = (1,)
        self.observation_dim = (2,)

        if self.max_torque > constants.TORQUE_MAX:
            print(f"Max torque specified larger than settings value {constants.TORQUE_MAX}. Changing to settings value.")
            self.max_torque = constants.TORQUE_MAX

    def connect(self):

        self.pbm = Manager.BEAR(port=self.port, baudrate=8000000)


        self._set_bear_parameters()

        # change to torque contol mode
        self.pbm.set_mode((self.motor_id, constants.TORQUE_MODE))
        self.pbm.set_goal_iq((self.motor_id, 0.0))
        # self.run_homing_routine()

    def step(self, torque):
        self._apply_torque(torque)
        x = self.observe()

        # TODO: sleep for dt?
        time.sleep(self.dt*0.8)

        cost = -self.get_cost(x, torque)
        done = self.is_done(cost)
        # if done:
            # cost += self.exit_reward

        return x, cost, done, {}

    def stop(self):
        self.pbm.set_torque_enable((self.motor_id, 0))
        self.pbm.set_mode((self.motor_id, constants.TORQUE_MODE))
        self.pbm.set_goal_iq((self.motor_id, 0.0))
        time.sleep(self.time_to_stabilize)

    def reset(self):
        num_frames = 300
        dt_per_frame = 0.01

        self.stop()
        self.pbm.set_mode((self.motor_id, constants.DIRECT_FORCE_MODE))
        self.pbm.set_limit_iq_max((self.motor_id, 2.2*constants.INV_MOTOR_CONST))
        self.pbm.set_torque_enable((self.motor_id, 1))


        current_position, _ = self.observe()
        initial_position = np.random.uniform(*self.theta0_range)

        via_points = np.linspace(current_position, initial_position, num=num_frames)
        for point in via_points:
            self.pbm.set_goal_position((self.motor_id, point*constants.RAD2ENC))
            time.sleep(dt_per_frame)

        time.sleep(self.time_to_stabilize)

        self.pbm.set_limit_iq_max((self.motor_id, self.max_torque*constants.INV_MOTOR_CONST))
        self.pbm.set_mode((self.motor_id, constants.TORQUE_MODE))

        return self.observe()

    def run_homing_routine(self):
        self.stop()
        raw_encoder_reading = 0.0
        for i in range(10):
            raw_encoder_reading += self.pbm.get_present_position(self.motor_id)[0]/10.0
            time.sleep(0.01)

        enc_range = constants.ENCODER_COUNTS

        # self.homing_offset = raw_encoder_reading
        # self.homing_offset = constants.ENC2RAD*((self.homing_offset + 0.5*enc_range) % (enc_range) - 0.5*enc_range)

        # print(f"Homing offset: {self.homing_offset}")
        current_encoder_offset = self.pbm.get_homing_offset(self.motor_id)[0]
        new_offset = current_encoder_offset - raw_encoder_reading + 0.5*enc_range
        new_offset = (new_offset + 0.5*enc_range) % (enc_range) - 0.5*enc_range
        self.pbm.set_homing_offset((self.motor_id, int(new_offset)))
        print(current_encoder_offset)
        print(f"Offset set to {new_offset}")
        print(self.pbm.get_homing_offset(self.motor_id)[0])

        self.pbm.set_torque_enable((self.motor_id, 1))
        self.pbm.set_goal_iq((self.motor_id, 0.0))


    def _set_bear_parameters(self):
        self.pbm.set_limit_id_max((self.motor_id, 0.0))
        self.pbm.set_limit_iq_max((self.motor_id, self.max_torque*constants.INV_MOTOR_CONST))
        self.pbm.set_limit_velocity_max((self.motor_id, self.max_speed))
        self.pbm.set_p_gain_position((self.motor_id, constants.POS_P))
        self.pbm.set_i_gain_position((self.motor_id, constants.POS_I))
        self.pbm.set_d_gain_position((self.motor_id, constants.POS_D))
        self.pbm.set_p_gain_velocity((self.motor_id, constants.VEL_P))
        self.pbm.set_i_gain_velocity((self.motor_id, constants.VEL_I))
        self.pbm.set_d_gain_velocity((self.motor_id, constants.VEL_D))
        self.pbm.set_p_gain_force((self.motor_id, constants.FOR_P))
        self.pbm.set_i_gain_force((self.motor_id, constants.FOR_I))
        self.pbm.set_d_gain_force((self.motor_id, constants.FOR_D))
        self.pbm.set_p_gain_iq((self.motor_id, constants.IQ_P))
        self.pbm.set_i_gain_iq((self.motor_id, constants.IQ_I))
        self.pbm.set_d_gain_iq((self.motor_id, constants.IQ_D))
        self.pbm.set_p_gain_id((self.motor_id, constants.ID_P))
        self.pbm.set_i_gain_id((self.motor_id, constants.ID_I))
        self.pbm.set_d_gain_id((self.motor_id, constants.ID_D))
        self.pbm.set_temp_limit_low((self.motor_id, 85.0))

        self.pbm.set_homing_offset((self.motor_id, int(self.encoder_offset)))

    def _apply_torque(self, torque):
        self.pbm.set_torque_enable((self.motor_id, 1))
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        velocity = self.pbm.get_present_velocity((self.motor_id))[0]*constants.ENC2RAD
        if np.abs(velocity) > self.max_speed:
            torque = 0.0

        # command motor torque
        self.pbm.set_goal_iq((self.motor_id, torque*constants.INV_MOTOR_CONST))

    def _subtract_homing_offset(self, angle):
        return (angle - self.homing_offset + np.pi) % (2*np.pi) - np.pi

    def observe(self):
        # read_registers = ['present_position', 'present_velocity']
        # error_status, motor_status, data = self.pbm.bulk_read([1], read_registers)
        # position = data[0][0]*constants.ENC2RAD
        # velocity = data[0][1]*constants.ENC2RAD

        position = self.pbm.get_present_position((self.motor_id))[0]*constants.ENC2RAD
        velocity = self.pbm.get_present_velocity((self.motor_id))[0]*constants.ENC2RAD

        # position = (position - self.homing_offset + np.pi) % (2*np.pi) - np.pi
        return np.array([position, velocity], dtype=np.float64)

    def get_cost(self, x, u):
        q_cost = np.matrix([x]) @ self.Q @ np.matrix([x]).T
        costs = q_cost + u @ self.R @ np.transpose(u)
        return costs.item()

    def is_done(self, cost):
        p, v = self.observe()
        done = True if np.abs(p) < self.pos_exit_threshold and np.abs(v) < self.vel_exit_threshold else False
        return done

    def get_state_matrices(self):
        A = np.matrix([[0, 1], [self.g / self.l, -self.b / (self.m * self.l ** 2)]], dtype=np.float64)
        B = np.matrix([[0], [1 / (self.m * self.l ** 2)]], dtype=np.float64)

        C = np.matrix([[1, 1]], dtype=np.float64)
        D = np.matrix([[0]], dtype=np.float64)

        return A, B, C, D

    def lqr_kp(self):
        A, B, C, D = self.get_state_matrices()

        if self.use_lqr_discrete:
            A, B, C, D, _ = cont2discrete((A, B, C, D), self.dt)
            P = solve_discrete_are(A, B, self.Q, self.R)
        else:
            P = solve_continuous_are(A, B, self.Q, self.R)

        K = self.R.I * B.T * P
        return K, P

if __name__ == '__main__':
    pass