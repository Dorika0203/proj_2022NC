import torch.nn as nn


class MyLinearNet(nn.Module):
    
    def __init__(self, in_features=512, out_features=512):
        
        super(MyLinearNet, self).__init__()
        
        self.l1 = nn.Linear(in_features=in_features, out_features=in_features*2)
        self.l2 = nn.Linear(in_features=2*in_features, out_features=2*out_features)
        self.l3 = nn.Linear(in_features=2*out_features, out_features=out_features)
        self.a1 = nn.ReLU()
        self.a2 = nn.ReLU()
        self.a3 = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.2)

    
    def forward(self, x):
        o = self.a1(self.l1(x))
        o = self.dropout(self.a2(self.l2(o)))
        o = self.a3(self.l3(x))
        return o



def MainModel(**kwargs):
    
    model = MyLinearNet(in_features=512, out_features=512)
    return model