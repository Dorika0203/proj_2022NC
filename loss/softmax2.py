import torch
import torch.nn as nn
from utils import accuracy


class LossFunction(nn.Module):

	def __init__(self, nOut, nClasses, **kwargs):

		super(LossFunction, self).__init__()

		self.criterion = torch.nn.CrossEntropyLoss()
		self.fc = nn.Linear(nOut, nClasses)

		saved_state_dict = torch.load('baseline_v2_ap.model')
		self.fc.weight = torch.nn.Parameter(saved_state_dict['__L__.softmax.fc.weight'])
		self.fc.bias = torch.nn.Parameter(saved_state_dict['__L__.softmax.fc.bias'])
		print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x = self.fc(x)
		nloss = self.criterion(x, label)
		prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1
