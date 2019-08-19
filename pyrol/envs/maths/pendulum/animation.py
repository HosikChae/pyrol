import numpy as np
import matplotlib.pyplot as plt
from pyrol.plots.matplotlib_ext import DynamicUpdater


class PendulumAnimation(DynamicUpdater):
    def __init__(self, theta=0., time=0., u=0., rod_length=1.):
        self.l = rod_length
        self.fig = plt.figure(figsize=(7., 7.))
        self.ax = self.fig.add_subplot(1, 1, 1)
        d = self.l + 0.25
        self.ax.set_xlim(-d, d)
        self.ax.set_ylim(-d, d)
        x, y, dx, dy, orientation, size = self.update_marker(theta, u)
        self.arrow = self.ax.plot([x, x + dx], [y, y + dy], 'r', marker=(3, 0, orientation), markersize=size)
        self.pendulum = self.ax.plot([0, x], [0, y], 'ko-', lw=2, markersize=18)
        self.time = self.ax.text(-d + 0.05, d - 0.1, f't   = {time:.2f} s')
        self.theta = self.ax.text(-d + 0.05, d - 0.175, f'th = {theta * 180 / np.pi:.2f} degrees')
        self.u = self.ax.text(-d + 0.05, d - 0.25, f'u  = {u:.2f} N')

    def close(self):
        plt.close()

    def update_marker(self, theta, u):
        x = self.l * np.sin(theta)
        y = self.l * np.cos(theta)
        dx = .1 * u * np.cos(theta)
        dy = - .1 * u * np.sin(theta)
        orientation = -theta * 180 / np.pi + np.sign(u) * -90
        size = 10. * np.absolute(u)
        return x, y, dx, dy, orientation, size

    def update(self, theta, time, u):
        self.time.set_text(f't   = {time:.2f} s')
        self.theta.set_text(f'th = {theta * 180 / np.pi:.2f} degrees')
        self.u.set_text(f'u  = {u:.2f} N')
        x, y, dx, dy, orientation, size = self.update_marker(theta, u)
        self.arrow[0].set_data([x, x + dx], [y, y + dy])
        self.arrow[0].set_marker((3, 0, orientation))
        self.arrow[0].set_markersize(size)
        self.pendulum[0].set_data([0, x], [0, y])
        self.fig.canvas.draw()

