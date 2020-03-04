import unittest
import numpy as np
from pyrol.buffers import ReplayBasic


class TestReplayBuffers(unittest.TestCase):
    def test_replay_basic(self):
        SIZE = 1000000
        FIELDS = 5
        MEAN = 0.0
        STDDEV = 1.0
        EPS = 0.001 * STDDEV

        buffer = ReplayBasic(capacity=SIZE)
        for sample in np.random.normal(MEAN, STDDEV, size=(SIZE, FIELDS)):
            buffer.push(*sample)

        samples = buffer.sample(SIZE)

        mu = np.sum(np.mean(samples, axis=1)) / FIELDS
        sig = np.sum(np.std(samples, axis=1)) / FIELDS

        self.assertLess(abs(mu - MEAN), EPS, msg=f'Replay buffer mean off by: {abs(mu - MEAN)}.')
        self.assertLess(abs(sig - STDDEV), EPS, msg=f'Replay buffer standard deviation off by: {abs(sig - STDDEV)}.')

