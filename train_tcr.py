from sklearn.model_selection import learning_curve
from zmq import device
from tcr_encopder import *
import torch
from torch import nn
device = torch.device("cuda:2")
encoder = TCRE()

learning_rate = 1e-4
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(encoder.parameters(),lr=learning_rate)