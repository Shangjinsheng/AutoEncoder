import torch
from torch import nn
from torch.nn import functional as F


class TCRE(nn.Module):
    def __init__(self):
        super(TCRE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels= 30,kernel_size=(5,2),stride= 1),#卷积1号
            nn.BatchNorm2d(30),
            nn.AvgPool2d((4,1)),#前面要有个BatchNormal
            nn.Conv2d(in_channels= 30,out_channels= 20,kernel_size= (4,2),stride= 1),#卷积2号
            nn.BatchNorm2d(20),
            nn.AvgPool2d((4,1)),#前面要有个BatchNormal
            nn.Flatten(),#展平
            nn.Linear(240,30),#线性层
            nn.Dropout(),
            nn.Linear(30,30),
            nn.Dropout(),
            nn.Linear(30,30),
            nn.Dropout(),
            nn.Linear(30,240)            
        )
        self.BN1 = nn.BatchNorm2d(20)
        self.BN2 =nn.BatchNorm2d(30)

        self.Conv3 = nn.Conv2d(20,30,(4,3),1)
        self.conv4 = nn.Conv2d(30,1,(6,4),1)
    def forward(self,x):
        x = self.encoder(x)
        x = x.reshape(-1,20,4,3)
        x =self.BN1(x)      
        x = F.interpolate(x,size=[20,6])        
        x = self.Conv3(x)
        x = self.BN2(x)
        x = F.interpolate(x,size=[85,8])
        x = self.conv4(x)
        return x
        
        
if __name__ == '__main__':
    model = TCRE()
    input = torch.ones((3,1,80,5))
    output = model(input)
    print(output.shape)
