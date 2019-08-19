import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from pyrol.utils.math_utils import remainder
from pyrol.envs.maths.pendulum.animation import PendulumAnimation


def _pendulum_model(y, t, m, g, l, b, u):
    th, th_dot = y
    dydt = [th_dot,
            (u - b * th_dot + m * g * l * np.sin(th)) / (m * l ** 2)]
    return dydt


class PendulumEnv(gym.Env):

    def __init__(self,
                 measurement_noise=0.,
                 actuator_noise=0.,
                 initialization_noise=0.,
                 length=1.,
                 damping=0.1,
                 mass=1.,
                 max_speed=10.,
                 max_torque=2.,
                 seed=100,
                 step_dt=0.005,
                 dynamics_resolution=100,
                 g=9.807,
                 th0=0.,
                 th_dot0=0.,
                 render=True,
                 ):
        self.measurement_noise = measurement_noise
        self.actuator_noise = actuator_noise
        self.initialization_noise = initialization_noise
        self.resolution = dynamics_resolution
        self.l = length
        self.m = mass
        self.b = damping
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.step_dt = step_dt
        self.g = g

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float64)
        obs_max = np.array([np.pi, self.max_speed])  # th, th_dot
        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float64)

        self.np_random = None
        self.seed(seed)
        self.th0 = th0
        self.th_dot0 = th_dot0
        self.state = np.array([self.th0, self.th_dot0, 0.], dtype=np.float64)  # th, th_dot, time
        self._init_state()  # Noisy starting position
        self.traj = np.array([[self.state[0].copy(), self.state[1].copy()]])

        self.do_render = render
        if self.do_render:
            self.animate = self.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _run_dynamics(self, u):
        y0 = [self.state[0], self.state[1]]
        t = np.linspace(self.state[2], self.state[2] + self.step_dt, self.resolution)
        sol = odeint(_pendulum_model, y0, t, args=(self.m, self.g, self.l, self.b, u))
        self.state[0], self.state[1] = sol[-1, :]
        if np.absolute(self.state[1]) >= self.max_speed:
            f'You\'ve reached max speed of {self.max_speed}!'
            self.state[1] = np.clip(self.state[1], -self.max_speed, self.max_speed)
        self.state[0] = remainder(self.state[0], (2 * np.pi))
        self.state[2] = self.state[2] + self.step_dt

        self.traj = np.concatenate((self.traj, sol[1:, :]))

        assert self.state[2] == t[-1]

    def apply_torque(self, u):
        u = u + np.random.normal(loc=0., scale=self.actuator_noise)
        u = np.clip(u, -self.max_torque, self.max_torque)
        self._run_dynamics(u)
        return None

    def _measure(self):
        th = self.state[0].copy() + np.random.normal(loc=0., scale=self.measurement_noise)
        th = remainder(th, (2 * np.pi))
        th_dot = self.state[1].copy() + np.random.normal(loc=0., scale=self.measurement_noise)
        return np.array([th, th_dot], dtype=np.float64)

    def step(self, u):
        self.apply_torque(u)
        x = np.absolute(self._measure())
        Q = np.matrix([[1.0, 0.], [0., 1.0]], dtype=np.float64)
        R = np.matrix([[1]], dtype=np.float64)
        costs = np.matrix([x]) @ Q @ np.matrix([x]).T + u @ R @ np.transpose(u)
        costs = costs.item()

        if self.do_render:
            self.animate.update(theta=self.state[0], time=self.state[2], u=u[0])

        return x, costs, False, {}

    def _init_state(self):
        th, th_dot = np.array([self.th0, self.th_dot0]) + np.random.normal(size=2, loc=0., scale=self.initialization_noise)
        self.state = np.array([th, th_dot, 0.0], dtype=np.float64)

    def reset(self):
        self._init_state()
        if self.do_render:
            self.animate.update(theta=self.state[0], time=self.state[2], u=0.)
        return self._measure()

    def render(self, mode='human'):
        return PendulumAnimation(theta=self.state[0], time=self.state[2], u=0.)

    def close(self):
        if self.do_render:
            self.animate.close()
        else:
            pass


# if __name__ =='__main__':
#     # Test 2: Test if the class in gazebo env works properly
#     pend = PendulumEnv(g=5., damping=0.25, step_dt=0.05, th0=0 - 0.01, max_torque=10.)
#
#     count = 0
#     while pend.state[2] < 10:
#         if count % 5 == 0:
#             u = -1.
#         else:
#             u = -1.
#         pend.step(np.array([u]))
#         count = count + 1
#     pend.close()
#     traj = pend.traj.copy()

    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), traj[:, 0], 'b', label='theta(t)')
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), [np.pi] * traj[:, 0].size, 'k', label='pi~3.14')
    # plt.plot(np.linspace(0, pend.state[2], traj.shape[0]), traj[:, 1], 'g', label='th_dot(t)')
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
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), trajectory[1:, 0], 'b', label='theta(t)')
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), [np.pi] * trajectory[1:, 0].size, 'k', label='pi~3.14')
    # plt.plot(np.linspace(0, time, trajectory.shape[0] - 1), trajectory[1:, 1], 'g', label='th_dot(t)')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show(block=True)

