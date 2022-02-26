from sklearn.model_selection import learning_curve
from zmq import device
import sys
sys.path.append("/home/shangjinsheng/AutoEncoder/tcr_encoder.py")
from tcr_encoder import *
import torch
from torch import nn
from data_processing import *
from torch.utils.data import Dataset,DataLoader

device = torch.device("cuda:2")
encoder = TCRE()

learning_rate = 1e-4
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(encoder.parameters(),lr=learning_rate)


class Mydataset(Dataset):
    def __init__(self,tcr_array):
        self.tcr_array = tcr_array
    
    def __getitem__(self,index):
        return self.tcr_array[index]
    
    def __len__(self):
        return len(self.tcr_array)
    

train_set = Mydataset(TCR_array)
train_loader = DataLoader(dataset =train_set,batch_size = 16,shuffle = True )
