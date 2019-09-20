import scipy
import numpy as np
import matplotlib.pyplot as plt

from pyrol.envs.maths.pendulum import PendulumEnv

SEED = 0
STEPS = 200
OFFSET = 45 * np.pi / 180

if __name__ == '__main__':
    env = PendulumEnv(th0=OFFSET, max_torque=0.07)
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    g = env.g
    l = env.l
    m = env.m
    b = env.b

    Q = env.Q
    R = env.R
    A = np.matrix([[0, 1], [g / l, -b / (m * l ** 2)]], dtype=np.float64)
    B = np.matrix([[0], [1 / (m * l ** 2)]], dtype=np.float64)

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = R.I * B.T * P

    for _ in range(STEPS):
        x = env.observe()
        u = -K @ x
        print(f'u: {u.item()}  |  th: {x[0]}  |  th_dot: {x[1]}')
        env.step(u)

    traj = env.traj

    plt.plot(np.linspace(0, env.state[2], traj.shape[0]), [-np.pi] * traj[:, 0].size, '--k', label='pi~3.14')
    plt.plot(np.linspace(0, env.state[2], traj.shape[0]), [np.pi] * traj[:, 0].size, '--k')
    plt.plot(np.linspace(0, env.state[2], traj.shape[0]), traj[:, 1], 'g', label='th_dot(t)')
    plt.plot(np.linspace(0, env.state[2], traj.shape[0]), traj[:, 0], 'b', label='theta(t)')
    plt.xlabel('t')
    plt.grid()
    plt.legend(loc='best')
    plt.show(block=True)


