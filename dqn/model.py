import torch.nn as nn
import torch.nn.functional as F

class BlockBlastNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(BlockBlastNet, self).__init__()

    def forward(self, x):
        return x