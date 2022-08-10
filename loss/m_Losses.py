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
        # elif self.trainfunc == 'MSE_SoftmaxProto':
        #     self.mse = nn.MSELoss()
        #     self.softmaxproto = softmaxproto2.LossFunction(**kwargs)
        # elif self.trainfunc == 'SoftmaxProto':
        #     self.softmaxproto = softmaxproto2.LossFunction(**kwargs)
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

        mult_emb, spk_label = y
        mult_emb = mult_emb.cuda()
        spk_label = spk_label.cuda()
        # breakpoint()

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

        elif self.trainfunc == 'MyTriplet_CS':
            # spk_label은 여기서 다른 화자의 다발화 임베딩이 됨
            loss = 1 - self.cs(x, mult_emb).mean() + \
                self.cs2(x, spk_label).mean()
        elif self.trainfunc == 'DA':
            # breakpoint()
            dist = 1 - self.cs(x[0::2], x[1::2])
            same_dist = dist[0::2].reshape(-1, 1)  # 100
            diff_dist = dist[1::2].reshape(-1, 1)  # 100
            
            ss = torch.cdist(same_dist, same_dist).flatten()
            tt = torch.cdist(diff_dist, diff_dist).flatten()
            ts = torch.cdist(same_dist, diff_dist).flatten()
            
            loss = torch.exp(-(ss**2)/2).mean() + torch.exp(-(tt**2)/2).mean() - 2*torch.exp(-(ts**2)/2).mean()
            
            # breakpoint()

        return loss, prec
