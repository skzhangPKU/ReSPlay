import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from config import COLUMN,ROW,VTR_OUTPUT_DIM,HIDDEN_DIM

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(VTR_OUTPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, COLUMN * ROW)
        self.relu = nn.Tanh()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

