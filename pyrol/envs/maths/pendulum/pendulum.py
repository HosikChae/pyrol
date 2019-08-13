import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint


def _pendulum_model(y, t, m, g, l, b, u, th_ddot0):
    th, th_dot = y
    dydt = [th_dot,
            (- b * th_dot - m * g * l * np.sin(th) + u) / (m * l ** 2) - th_ddot0]
    return dydt


if __name__ =='__main__':
    m = 1.
    g = 9.8
    l = 1.
    b = 0.01
    u = np.zeros(200)
    u[0] = 0.
    dt = 0.5
    y0 = [np.pi, 0.]
    th_ddot0 = 0.
    time = 0.
    trajectory = np.zeros((1, 2))
    for i in range(u.size):
        t = np.linspace(i * dt, dt, 10)
        sol = odeint(_pendulum_model, y0, t, args=(m, g, l, b, u[i], th_ddot0))
        th = sol[-1, 0]
        th_dot = sol[-1, 1]
        y0 = [th, th_dot]
        th_ddot0 = (- b * th_dot - m * g * l * np.sin(th) + u[i]) / (m * l ** 2)
        # th_ddot0 = 0
        trajectory = np.concatenate((trajectory, sol[1:]))
        time = time + dt

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkagg')

    plt.plot(np.linspace(0, time, trajectory.shape[0]), trajectory[:, 0], 'b', label='theta(t)')
    plt.plot(np.linspace(0, time, trajectory.shape[0]), trajectory[:, 1], 'g', label='th_dot(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


class PendulumEnv(gym.Env):

    def __init__(self,
                 measurement_noise=0.,
                 actuator_noise=0.,
                 length=1.,
                 damping=0.1,
                 mass=1.,
                 max_speed=10.,
                 max_torque=2.,
                 seed=100,
                 step_dt=0.005,
                 g=9.807
                 ):
        self.measurement_noise = measurement_noise
        self.actuator_noise = actuator_noise
        self.l = length
        self.m = mass
        self.b = damping
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.step_dt = step_dt
        self.g = g

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float64)
        obs_max = np.array([1., 1., self.max_speed])
        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float64)

        self.np_random = None
        self.seed(seed)
        self.state = np.zeros(4, dtype=np.float64)  # th, th_dot, th_ddot, time

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _run_dynamics(self, u):
        _pendulum_model(u)
        # TODO: complete this function and add noisy measurements

    def apply_torque(self, u):
        u = u + np.random.normal(loc=0., scale=self.actuator_noise)
        u = np.clip(u, -self.max_torque, self.max_torque)
        self._run_dynamics(u)
        return None

    def step(self, u):
        self.apply_torque(u)

        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self):
        pass

    def close(self):
        pass

