import torch.nn as nn
import importlib
import loss.m_Losses

class EmbedNet(nn.Module):
    # SpeakerNet: Model과 Loss Function을 같이 묶어서 처리함.
    
    def __init__(self, model, **kwargs):
        super(EmbedNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        # LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = loss.m_Losses.LossFunction(**kwargs)

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            label = label.cuda()
            nloss = self.__L__.forward(outp, label)
            return nloss
        