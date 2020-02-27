import time

from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

from utils import remainder
from .animation import PendulumAnimation
from .math_env import Env

class PendulumEnv(Env):
    def __init__(self,
                 max_speed=100.0,
                 max_torque=0.1,  
                 step_dt=0.05,
                 th0=0.,  
                 th_dot0=0.,
                 th_init=(-2*0.1, 2*0.1),  # (-np.pi, np.pi)
                 thdot_init=(-0.01, 0.01),  # (-0.095, 0.095)
                 exit_reward=50,
                 pos_exit_threshold=0.2,
                 vel_exit_threshold=0.3,
                 l=0.37,
                 m=0.4,
                 b=0.1,
                 g =9.8907,
                 ):

        # Pendulum parameters
        self.l = l
        self.m = m
        self.b = b
        self.g = g

        self.Q = np.matrix([[1.0, 0.], [0., 0.1]], dtype=np.float64)
        self.R = np.matrix([[.001]], dtype=np.float64)
        
        # Noise parameters
        self.torque_noise = 0.0
        self.th_obs_noise = 0.0
        self.thdot_obs_noise = 0.0

        # Init parameters
        self.th_range=th_init  
        self.thdot_range=thdot_init

        self.step_dt = step_dt
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.t = np.linspace(0, self.step_dt, 2)

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float64)
        obs_max = np.array([2 * np.pi, self.max_speed])  # th, th_dot
        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float64)

        self.done_reward = exit_reward
        self.pos_exit_threshold = pos_exit_threshold
        self.vel_exit_threshold = vel_exit_threshold

        self.th0 = th0
        self.th_dot0 = th_dot0
        self.state = np.array([self.th0, self.th_dot0, self.t[0]], dtype=np.float64)
        self.traj = np.array([[self.state[0].copy(), self.state[1].copy()]])

        self.do_render = False
        self.np_random = None

    # Main Methods
    def step(self, u):
        self.apply_torque(u)
        x = self.observe()
        cost = -self.get_cost(x, u)

        if self.do_render:
            self.animate.update(theta=self.state[0], time=self.state[2], u=u[0])
        
        done = self.is_done(cost)
        if done:
            cost += self.done_reward

        return x, cost, done, {}

    def apply_torque(self, u):
        u = u + np.random.normal(loc=0., scale=self.torque_noise)
        u = np.clip(u, -self.max_torque, self.max_torque)
        self._run_dynamics(u)

    def _run_dynamics(self, u):
        self.state[0], self.state[1] = self.euler_propagation(self.state[0], self.state[1], u)
        if np.absolute(self.state[1]) >= self.max_speed:
            self.state[1] = np.clip(self.state[1], -self.max_speed, self.max_speed)
        self.state[0] = self.calc_th(self.state[0])
        self.state[2] = self.state[2] + self.step_dt
        self.traj = np.concatenate((self.traj, np.array([[self.state[0].copy(), self.state[1].copy()]])))

    def calc_th(self, th, y=2*np.pi):
        th = remainder(th.copy(), y)
        if th < -np.pi:
            th += y
        elif th > np.pi:
            th -= y
        th = -th if th == -np.pi else th
        return th

    def observe(self):
        th = self.state[0].copy() + np.random.normal(loc=0., scale=self.th_obs_noise)
        th = self.calc_th(th)
        th_dot = self.state[1].copy() + np.random.normal(loc=0., scale=self.thdot_obs_noise)
        return np.array([th, th_dot], dtype=np.float64)

    def get_cost(self, x, u):
        q_cost = np.matrix([x]) @ self.Q @ np.matrix([x]).T
        costs = q_cost + u @ self.R @ np.transpose(u)
        return costs.item()

    def is_done(self, cost):
        p, v = self.observe()
        if np.abs(p) < self.pos_exit_threshold and np.abs(v) < self.vel_exit_threshold:
            return True
        else:
            return False

    # Animation Methods
    def render(self, mode='human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            self.do_render = True
            plt.ion()
            self.animate = PendulumAnimation(theta=self.state[0].copy(), time=self.state[2].copy(), u=0.)

    def close(self):
        if self.do_render:
            self.animate.close()
        else:
            pass

    # Reset Methods
    def reset(self):
        self._init_state(th=self.th_range, thdot=self.thdot_range)
        if self.do_render:
            self.animate.update(theta=self.state[0].copy(), time=self.state[2].copy(), u=0.)
        return self.observe()

    def _init_state(self, th=(-np.pi, np.pi), thdot=(-0.095, 0.095)):
        self.state[0] = self.np_random.uniform(*th) if self.np_random else np.random.uniform(*th)
        self.state[1] = self.np_random.uniform(*thdot) if self.np_random else np.random.uniform(*thdot)
        self.state[2] = 0.0

    # Naive way to propagate dynamics but slightly faster
    def euler_propagation(self, th, thdot, u, res=5):
        th = th.copy()
        thdot = thdot.copy()
        u = u.item()
        dt = self.step_dt/res
        for _ in range(res):
            th0 = th
            th += dt*thdot
            thdot += dt*(u - self.b*thdot + self.m*self.l*self.g*np.sin(th0))/(self.m*self.l**2)
        return [th, thdot]

    # Data Augmentation Methods
    def symmetry_func_dict(self):
        multiplier = 2
        return ({'reflection': self.reflection,}, multiplier)

    def reflection(self, state, action, next_state, reward, done):
        """Note: Range should be symmetric around goal point--if
        you calculate cost without symmetry will get different cost
        with LQR. Therefore, when there is a non-trivial reward 
        function or transformation and resulting reward then
        it will be very difficult to figure out exactly what reward.
        """
        s = -state.copy()
        s[0] = self.calc_th(s[0])
        a = -action.copy()
        next_s = -next_state.copy()
        next_s[0] = self.calc_th(next_s[0])
        r = -self.get_cost(next_s, a)
        return (s, a, next_s, r, done)

    def get_state_matrices(self):
        A = np.matrix([[0, 1], [self.g / self.l, -self.b / (self.m * self.l ** 2)]], dtype=np.float64)
        B = np.matrix([[0], [1 / (self.m * self.l ** 2)]], dtype=np.float64)

        C = np.matrix([[1, 1]], dtype=np.float64)
        D = np.matrix([[0]], dtype=np.float64)

        return A, B, C, D


    # LQR
    def lqr_kp(self):
        A, B, _, _ = self.get_state_matrices()

        P = solve_continuous_are(A, B, self.Q, self.R)
        K = self.R.I * B.T * P
        return K, P

    # Misc
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
        