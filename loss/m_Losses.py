import torch.nn as nn
import loss.softmaxproto2 as softmaxproto2
import loss.softmax2 as softmax2
import torch


class LossFunction(nn.Module):
    def __init__(self, trainfunc, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = False
        self.trainfunc = trainfunc

        if self.trainfunc == 'MSE':
            self.mse = nn.MSELoss()
        elif self.trainfunc == 'MAE':
            self.mae = nn.L1Loss()
        elif self.trainfunc == 'CS':
            self.test_normalize = True
            self.cs = nn.CosineSimilarity()
        elif self.trainfunc == 'MSE_CS':
            self.test_normalize = True
            self.mse = nn.MSELoss()
            self.cs = nn.CosineSimilarity()
        elif self.trainfunc == 'MSE_Softmax':
            self.mse = nn.MSELoss()
            self.softmax = softmax2.LossFunction(**kwargs)
        elif self.trainfunc == 'Softmax':
            self.softmax = softmax2.LossFunction(**kwargs)
        elif self.trainfunc == 'MyTriplet_CS':
            self.test_normalize = True
            self.cs = nn.CosineSimilarity()
            self.cs2 = nn.CosineSimilarity()
        elif self.trainfunc == 'DA':
            self.test_normalize = True
            self.cs = nn.CosineSimilarity()
        else:
            raise ValueError('No Loss function - {}'.format(self.trainfunc))

    def forward(self, x, y):

        loss = 0
        prec = None

        if self.trainfunc == 'MSE':
            mult_emb, _ = y
            mult_emb = mult_emb.cuda()
            
            loss = self.mse(x, mult_emb)
            
        elif self.trainfunc == 'MAE':
            mult_emb, _ = y
            mult_emb = mult_emb.cuda()
            
            loss = self.mae(x, mult_emb)
            
        elif self.trainfunc == 'CS':
            mult_emb, _ = y
            breakpoint()
            mult_emb = mult_emb.cuda()
            
            loss = 1 - self.cs(x, mult_emb).mean()
        elif self.trainfunc == 'MSE_CS':
            mult_emb, _ = y
            mult_emb = mult_emb.cuda()
            
            loss = 1 - self.cs(x, mult_emb).mean() + self.mse(x, mult_emb)
            
        elif self.trainfunc == 'MSE_Softmax':
            mult_emb, spk_label = y
            mult_emb = mult_emb.cuda()
            spk_label = spk_label.cuda()
            
            loss, prec = self.softmax(x, spk_label)
            loss += self.mse(x, mult_emb)
            
        elif self.trainfunc == 'Softmax':
            _ , spk_label = y
            spk_label = spk_label.cuda()
            
            loss, prec = self.softmax(x, spk_label)

        elif self.trainfunc == 'MyTriplet_CS':
            
            mult_emb, diff_spk_mult_emb = y
            mult_emb = mult_emb.cuda()
            diff_spk_mult_emb = diff_spk_mult_emb.cuda()            
            
            loss = 1 - self.cs(x, mult_emb).mean() + self.cs2(x, diff_spk_mult_emb).mean()
            
        elif self.trainfunc == 'DA':
            
            feat, cut_idx = x
            mult_emb, _ = y
            mult_emb = mult_emb.cuda()
            
            same_feat = feat[0:cut_idx]
            diff_feat = feat[cut_idx:]
            same_dist = 1-self.cs(same_feat[0::2], same_feat[1::2])
            diff_dist = 1-self.cs(diff_feat[0::2], diff_feat[1::2])
            same_dist = torch.unsqueeze(same_dist, dim=1)
            diff_dist = torch.unsqueeze(diff_dist, dim=1)
            
            ss = torch.cdist(same_dist, same_dist).flatten()
            tt = torch.cdist(diff_dist, diff_dist).flatten()
            ts = torch.cdist(same_dist, diff_dist).flatten()
            
            loss = torch.exp(-(ss**2)/2).mean() + torch.exp(-(tt**2)/2).mean() - 2*torch.exp(-(ts**2)/2).mean()
            # USE Precision as MMD Loss observer (originally used for classification accuracy)
            prec = loss.clone().detach()
            loss += 1 - self.cs(feat, mult_emb[0]).mean()

        return loss, prec
