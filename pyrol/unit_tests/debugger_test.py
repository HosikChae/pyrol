import unittest
import torch
import torch.nn as nn
from pyrol.utils.metrics import is_cuda
from pyrol.approximators.nn import Actor


class TestDebugger(unittest.TestCase):
    def test_is_cuda(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bool = True if device.type == 'cuda' else False
        case1 = torch.tensor([1]).to(device)
        case2 = torch.tensor([1])
        case3 = Actor(3, 2, 1).to(device)
        case4 = Actor(3, 2, 1)
        self.assertIs(is_cuda(case1), bool)
        self.assertIs(is_cuda(case2), False)
        self.assertIs(is_cuda(case3), bool)
        self.assertIs(is_cuda(case4), False)

