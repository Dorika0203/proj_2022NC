import torch.nn as nn


# class MyLinearNet(nn.Module):
    
#     def __init__(self, in_features=512, out_features=512):
        
#         super(MyLinearNet, self).__init__()
        
#         self.l1 = nn.Linear(in_features=in_features, out_features=in_features*2)
#         self.l2 = nn.Linear(in_features=2*in_features, out_features=2*out_features)
#         self.l3 = nn.Linear(in_features=2*out_features, out_features=out_features)
#         self.a1 = nn.ReLU()
#         self.a2 = nn.ReLU()
#         self.a3 = nn.ReLU()
#         self.dropout = nn.Dropout1d(p=0.2, inplace=False)

    
#     def forward(self, x):
#         o = self.a1(self.l1(x))
#         o = self.dropout(self.a2(self.l2(o)))
#         o = self.a3(self.l3(o))
#         return o


class MyLinearNet(nn.Module):
    
    def __init__(self, in_features=512, out_features=512):
        
        super(MyLinearNet, self).__init__()
        
        self.l1 = nn.Linear(in_features=in_features, out_features=in_features*2)
        self.l2 = nn.Linear(in_features=2*in_features, out_features=4*in_features)
        self.l3 = nn.Linear(in_features=4*in_features, out_features=4*out_features)
        self.l4 = nn.Linear(in_features=4*out_features, out_features=2*out_features)
        self.l5 = nn.Linear(in_features=2*out_features, out_features=out_features)
        
        self.a1 = nn.ReLU()
        self.a2 = nn.ReLU()
        self.a3 = nn.ReLU()
        self.a4 = nn.ReLU()
        self.a5 = nn.ReLU()
        
        self.dropout2 = nn.Dropout1d(p=0.2, inplace=False)
        self.dropout3 = nn.Dropout1d(p=0.2, inplace=False)
        self.dropout4 = nn.Dropout1d(p=0.2, inplace=False)
    
    def forward(self, x):
        o = self.a1(self.l1(x))
        o = self.dropout2(self.a2(self.l2(o)))
        o = self.dropout3(self.a3(self.l3(o)))
        o = self.dropout4(self.a4(self.l4(o)))
        o = self.a5(self.l5(o))
        return o


def MainModel(**kwargs):
    
    model = MyLinearNet(in_features=512, out_features=512)
    return model