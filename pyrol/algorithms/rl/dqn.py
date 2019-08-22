import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, q_model):
        super(DQN, self).__init__()
        self.q_model = q_model