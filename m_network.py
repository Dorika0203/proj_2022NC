import torch.nn as nn
import torch
import importlib

class EmbedNet(nn.Module):
    # SpeakerNet: Model과 Loss Function을 같이 묶어서 처리함.
    
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(EmbedNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)
            return nloss, prec1
        