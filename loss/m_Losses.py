import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self, trainfunc, **kwargs):
        super(LossFunction, self).__init__()

        self.trainfunc = trainfunc
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cs = nn.CosineSimilarity(dim=0)

    def forward(self, x, y):
        
        if self.trainfunc == 'MSE':
            return self.mse(x, y)
        elif self.trainfunc == 'MAE':
            return self.mae(x, y)
        elif self.trainfunc == 'CS':
            return self.cs(x, y)
        elif self.trainfunc == 'MSE_CS':
            return self.cs(x, y) + self.mse(x, y)
        else:
            raise ValueError('No Loss function - {}'.format(self.trainfunc))