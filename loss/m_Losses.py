import torch.nn as nn
import loss.softmaxproto2 as softmaxproto2
import loss.softmax2 as softmax
import torch

class LossFunction(nn.Module):
    def __init__(self, trainfunc, **kwargs):
        super(LossFunction, self).__init__()
        
        self.test_normalize = False
        self.trainfunc = trainfunc
        
        if trainfunc == 'CS' or trainfunc == 'MSE_CS':
            self.test_normalize = True
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cs = nn.CosineSimilarity(dim=0)
        # self.softmaxproto = softmaxproto2.LossFunction(**kwargs)
        self.softmax = softmax.LossFunction(**kwargs)

    def forward(self, x, y):
        
        mult_emb, spk_label = y
        mult_emb = mult_emb.cuda()
        spk_label = spk_label.cuda()
        
        loss = 0
        prec = None
        
        if self.trainfunc == 'MSE':
            loss = self.mse(x, mult_emb)
            
        elif self.trainfunc == 'MAE':
            loss = self.mae(x, mult_emb)
            
        elif self.trainfunc == 'CS':
            loss = 1 - self.cs(x, mult_emb).mean()
            
        elif self.trainfunc == 'MSE_CS':
            loss = 1 - self.cs(x, mult_emb).mean() + self.mse(x, mult_emb)
            
        elif self.trainfunc == 'MSE_Softmax':
            loss, prec = self.softmax(x, spk_label)
            loss += self.mse(x, mult_emb)
            
        elif self.trainfunc == 'Softmax':
            loss, prec = self.softmax(x, spk_label)
            
        # elif self.trainfunc == 'MSE_SoftmaxProto':
        #     loss, prec = self.softmaxproto(x, spk_label)
        #     loss += self.mse(x, mult_emb)
            
        # elif self.trainfunc == 'SoftmaxProto':
        #     loss, prec = self.softmaxproto(x, spk_label)
            
        else:
            raise ValueError('No Loss function - {}'.format(self.trainfunc))
        
        return loss, prec