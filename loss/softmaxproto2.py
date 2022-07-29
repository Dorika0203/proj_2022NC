#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.softmax as softmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        
        self.softmax = softmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)
        
        # copy last fc and ap loss parameters from baseline_v2_ap model
        saved_state_dict = torch.load('baseline_v2_ap.model')
        self.softmax.fc.weight = torch.nn.Parameter(saved_state_dict['__L__.softmax.fc.weight'])
        self.softmax.fc.bias = torch.nn.Parameter(saved_state_dict['__L__.softmax.fc.bias'])
        self.angleproto.w = torch.nn.Parameter(saved_state_dict['__L__.angleproto.w'])
        self.angleproto.b = torch.nn.Parameter(saved_state_dict['__L__.angleproto.b'])
        
        for param in self.softmax.parameters():
            param.requires_grad = False
        for param in self.angleproto.parameters():
            param.requires_grad = False
        
        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1