import torch.nn as nn
from models.uiencoder import Generator

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.generator = Generator()

    def forward(self,x):
        position = self.generator(x)
        return position