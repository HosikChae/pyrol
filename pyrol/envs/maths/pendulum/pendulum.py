import time
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

from pyrol.utils.maths import remainder
from pyrol.envs.maths.pendulum.animation import PendulumAnimation
from pyrol.envs.maths import Env
from pyrol.envs.real.bear_pendulum.bear_pendulum import BearPendulumEnv


class PendulumEnv(Env):
    def __init__(self,
                 max_speed=100.0,
                 max_torque=0.1,  # 0.8 underactuated
                 step_dt=0.05,
                 th0=0.,  # np.pi starts on bottom
                 th_dot0=0.,
                 th_init=(-2*0.1, 2*0.1),  # (-np.pi, np.pi)
                 thdot_init=(-0.01, 0.01),  # (-0.095, 0.095)
                 exit_reward=50,
                 use_bear_parameters=True,
                 pos_exit_threshold=0.2,
                 vel_exit_threshold=0.3,
                 l=0.37,
                 m=0.4,
                 b=0.1,
                 g =9.8907,
                 ):

        self.use_bear_parameters = use_bear_parameters
        self.bear_env = None
        if use_bear_parameters:
            # BEAR actuator (real world) paramters
            self.bear_env = BearPendulumEnv(theta0_range=th_init,
                                            max_speed=max_speed,
                                            max_torque=max_torque,
                                            dt=step_dt,
                                            exit_reward=exit_reward)
            self.l = self.bear_env.l
            self.m = self.bear_env.m
            self.g = self.bear_env.g
            self.b = self.bear_env.b

        else:
            # Pendulum parameters
            self.l = l
            self.m = m
            self.b = b
            self.g = g

        # Cost weighting
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

    def step(self, u):
        # Take a step in the environment
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
        # Using Euler method--poor approximation but fast
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

    def render(self, mode='human'):
        # animation
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

    def reset(self):
        self._init_state(th=self.th_range, thdot=self.thdot_range)
        if self.do_render:
            self.animate.update(theta=self.state[0].copy(), time=self.state[2].copy(), u=0.)
        return self.observe()

    def _init_state(self, th=(-np.pi, np.pi), thdot=(-0.095, 0.095)):
        self.state[0] = self.np_random.uniform(*th) if self.np_random else np.random.uniform(*th)
        self.state[1] = self.np_random.uniform(*thdot) if self.np_random else np.random.uniform(*thdot)
        self.state[2] = 0.0

    def euler_propagation(self, th, thdot, u, res=5):
        # propagate approximate dynamics
        th = th.copy()
        thdot = thdot.copy()
        u = u.item()
        dt = self.step_dt/res
        for _ in range(res):
            th0 = th
            th += dt*thdot
            thdot += dt*(u - self.b*thdot + self.m*self.l*self.g*np.sin(th0))/(self.m*self.l**2)
        return [th, thdot]

    def symmetry_func_dict(self):
        # for data augmentation to be used
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


    def lqr_kp(self):
        # LQR and gains
        if self.bear_env is not None:
            return self.bear_env.lqr_kp()
        else:
            A, B, _, _ = self.get_state_matrices()

            P = solve_continuous_are(A, B, self.Q, self.R)
            K = self.R.I * B.T * P
            return K, P

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

if __name__ == '__main__':
    # Test 2: Test if the class in gazebo env works properly
    # pend = PendulumEnv(g=9.8, damping=0.005, step_dt=0.05, th0=(0 - 0.01), max_torque=10.)
    max_torque = 0.8
    pend = PendulumEnv(th_init=(-30*np.pi/180,30*np.pi/180), thdot_init=(-0.01, 0.01), max_torque=max_torque, step_dt=0.01, use_bear_parameters=False)
    # pend = PendulumEnv(step_dt=0.05, th0=(0 + 0.1), max_torque=0.1)

    # pend.render(mode='human')
    # pend.reset()
    # count = 0
    # while pend.state[2] < 10:
    #     if count == 0:
    #         u = -0.0
    #     else:
    #         u = 0.0
    #     u = -max_torque
    #     pend.step(np.array([u]))
    #     count += 1
    # pend.close()
    # traj = pend.traj.copy()

    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), [-np.pi] * traj[:, 0].size, '--k', label='pi~3.14')
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), [np.pi] * traj[:, 0].size, '--k')
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), traj[:, 1], 'g', label='th_dot(t)')
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), traj[:, 0], 'b', label='theta(t)')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show(block=True)

    # # Test 1: test pendulum dynamics
    # m = 1.
    # g = 5.  # 9.8
    # l = 1.
    # b = 0.25
    # u = np.zeros(300)
    # u[0] = 0.
    # u[99] = 0.
    # u[199] = 0.
    # dt = 0.05
    # y0 = [0. + .001, 0.]
    # th_ddot0 = 0.
    # time = 0.
    # trajectory = np.array([y0])
    # for i in range(u.size):
    #     t = np.linspace(dt * i, dt * (i + 1), 10)
    #     sol = odeint(_pendulum_model, y0, t, args=(m, g, l, b, u[i]))
    #     th = remainder(sol[-1, 0], (2 * np.pi))
    #     th_dot = sol[-1, 1]
    #     y0 = [th, th_dot]
    #     trajectory = np.concatenate((trajectory, sol[1:]))
    #     time = time + dt
    
    # import matplotlib.pyplot as plt
    
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), trajectory[1:, 0], 'b', label='theta(t)')
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), [np.pi] * trajectory[1:, 0].size, 'k', label='pi~3.14')
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), trajectory[1:, 1], 'g', label='th_dot(t)')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show(block=True)

    # Test 3
    K, _ = pend.lqr_kp()
    K = np.array([[3.58891436, 1.86675319]])
    print(K)
    speeds = []


    trials = 10
    pend.render(mode='human')
    for _ in range(trials):
        count = 0
        x = pend.reset()
        done = False
        while not done and count < 300:
            u = -K @ x
            u = u.clip(-max_torque,max_torque)
            x, _, done, _ = pend.step(np.array([u]).flatten())
            count += 1
            speeds.append(x[1])

        print('Done')

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(speeds)
    plt.ioff()
    plt.show()

    pend.close()

