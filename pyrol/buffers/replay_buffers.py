import random
import numpy as np
from collections import namedtuple

from pyrol.utils.pytorch import return_self


class Base(object):
    r"""Replay Buffer Base Class
    Replay buffers can be used to store memory of transition states that can be used for training
    on. In the reinforcement learning a transition state is: ('state', 'action', 'next state', reward)

    API
        push: func, appends the transition state into memory
        sample: func, sample from memory to train from
        __len__: attr, number of transition states in memory buffer
    """

    def push(self, *args, **kwargs):
        """Push memory into replay buffer"""
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample from the replay buffer"""
        raise NotImplementedError

    def __len__(self):
        """Length or number of transition states currently in buffer"""
        raise NotImplementedError


class ReplayBasic(Base):
    r"""Basic implementation of a replay buffer"""
    def __init__(self, capacity, conversion=return_self, fields=('state', 'action', 'next_state', 'reward', 'done')):
        self.Transition = namedtuple('Transition', fields)
        self.capacity = capacity
        self.type_conversion = conversion
        self.memory = [None] * self.capacity
        self.position = 0

    def push(self, *args):
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        for replay in batch:
            self.push(replay)

    def sample(self, batch_size):
        transitions = []
        for field_data in [*zip(*random.choices(list(filter(None, self.memory)), k=batch_size))]:
            transitions.append(self.type_conversion(np.stack(field_data)))
        return transitions

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
        self.position = 0


class TorchReplayBasic(ReplayBasic):
    r"""Basic implementation of a replay buffer for PyTorch specific environments"""
    def __init__(self,
                 capacity,
                 conversion=return_self,
                 device='cpu',
                 fields=('state', 'action', 'next_state', 'reward', 'done')):
        super().__init__(capacity, conversion, fields)
        self.device = device

    def sample(self, batch_size):
        transitions = []
        for field_data in [*zip(*random.choices(list(filter(None, self.memory)), k=batch_size))]:
            transitions.append(self.type_conversion(np.stack(field_data)).to(self.device))
        return transitions

