#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):
        
        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        '''
        nPerSpeaker가 n이라 하면 한 speaker에 대해 n개의 다른 데이터에서 임베딩 벡터를 생성한다.
        이 때, 0번째 임베딩 벡터를 타겟, 1번째부터 n-1번째 까지의 평균을 앵커로 활용한다.
        그러면 총 200 * 2 만큼의 임베딩 벡터가 생기게 된다. batch size: 200, shape: (200,2,512)
        
        이 때 [k][0] - ([0][1], [1][1], [2][1], .., [B-1][1]) 까지 코사인 유사도를 구한다. (k : 0 ~ 199)
        그러면 총 200 x 200의 코사인 유사도 행렬이 구해진다.
        이 코사인 유사도 행렬은 단위행렬이 되어야 이상적이다.
        따라서 cross Entropy를 계산할 때, label 값으로 0, 1, 2, ..., 199를 주게 된다. (0번째 벡터는 0번째가 1, 1번째 벡터는 1번째가 1, ...)
        '''
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(cos_sim_matrix, label)
        prec1   = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1