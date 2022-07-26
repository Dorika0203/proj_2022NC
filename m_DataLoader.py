import os
import numpy
import torch
import warnings
import torch.distributed as dist
import soundfile
import random

from torch.utils.data import Dataset

warnings.simplefilter("ignore")

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

# 끝
class MyDataset(Dataset):
    
    def __init__(self, data_list, data_path, **kwargs):

        self.data_list = data_list
        
        # Read training files
        with open(data_list) as dataset_file:
            lines = dataset_file.readlines()

        # 라벨 데이터 (다 발화 임베딩 벡터) 딕셔너리로 정리
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        spk_dict = {}
        for spk in dictkeys:
            spk_dict[spk] = torch.FloatTensor(numpy.load(data_path+spk+'.npy'))

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = spk_dict[data[0]]
            filename = os.path.join(data_path,data[1][:-3]+'npy')
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)
        
        # # ONLY FOR TESTING CODE
        # if len(self.data_list) > 1000000:
        #     self.data_label = self.data_label[:len(self.data_label)//10]
        #     self.data_list = self.data_list[:len(self.data_list)//10]
    
    def __getitem__(self, index):

        dat = numpy.load(self.data_list[index])
        
        # frame by 평균, 후 L2 Normalization
        dat = torch.FloatTensor(dat)
        dat = torch.mean(dat, dim=0)
        dat = torch.nn.functional.normalize(dat, p=2, dim=0)
        
        return dat, self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class MyTestDataset(Dataset):
    def __init__(self, test_list, test_path, **kwargs):
        
        self.test_path  = test_path
        self.test_list = [i[:-3]+'npy' for i in test_list]

    def __getitem__(self, index):
        embed = numpy.load(os.path.join(self.test_path,self.test_list[index]))
        
        # frame by 평균, 후 L2 Normalization
        embed = torch.FloatTensor(embed)
        embed = torch.mean(embed, dim=0)
        embed = torch.nn.functional.normalize(embed, p=2, dim=0)
        
        # audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return embed, self.test_list[index]

    def __len__(self):
        return len(self.test_list)