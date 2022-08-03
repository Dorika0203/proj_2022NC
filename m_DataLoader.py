import os
import numpy
import torch
import warnings
import torch.distributed as dist

from torch.utils.data import Dataset

warnings.simplefilter("ignore")

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)
    
def round_down(num, divisor):
    return num - (num%divisor)

# 끝
class MyDataset(Dataset):
    
    def __init__(self, data_list, data_path, nPerSpeaker, multiple_embedding_flag, **kwargs):
        
        assert multiple_embedding_flag == 'B' or multiple_embedding_flag == 'C'
        
        self.sing_emb_list = data_list
        self.nPerSpeaker = nPerSpeaker
        
        # Read training files
        with open(data_list) as dataset_file:
            lines = dataset_file.readlines()

        # 라벨 데이터 (다 발화 임베딩 벡터) 딕셔너리로 정리
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        key2label = { key : ii for ii, key in enumerate(dictkeys) }
        spk_dict = {}
        if multiple_embedding_flag == 'B':
            for spk in dictkeys:
                spk_dict[spk] = torch.FloatTensor(numpy.load(data_path+spk+'.npy'))
        else:
            for spk in dictkeys:
                spk_dict[spk] = torch.FloatTensor(numpy.load(data_path+spk+'_type2.npy'))



        # Parse the training list into file names and ID indices
        self.sing_emb_list  = []
        self.mult_emb_label = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            mult_emb_label = spk_dict[data[0]]
            data_label = key2label[data[0]]
            filename = os.path.join(data_path,data[1][:-3]+'npy')
            
            self.mult_emb_label.append(mult_emb_label)
            self.sing_emb_list.append(filename)
            self.data_label.append(data_label)



        # ONLY FOR DEBUGGING
        label_counter_dict = {}
        select_idx_list = []
        for idx, item in enumerate(self.data_label):
            tmp = label_counter_dict.get(item, 0)
            # 각 label 별 10개씩만 뽑기
            if tmp > 10:
                continue
            tmp += 1
            label_counter_dict[item] = tmp
            select_idx_list.append(idx)
        
        self.sing_emb_list = [self.sing_emb_list[sel_idx] for sel_idx in select_idx_list]
        self.mult_emb_label = [self.mult_emb_label[sel_idx] for sel_idx in select_idx_list]
        self.data_label = [self.data_label[sel_idx] for sel_idx in select_idx_list]
            
    
    def __getitem__(self, index):
        if self.nPerSpeaker == 1:
            return self.single_getitem(index)
        else:
            return self.multiple_getitem(index)
    
    
    # 한개씩 sampling 되는 경우 사용
    def single_getitem(self, index):

        sing_emb = numpy.load(self.sing_emb_list[index])
        
        # frame by 평균, 후 L2 Normalization
        sing_emb = torch.FloatTensor(sing_emb)
        sing_emb = torch.mean(sing_emb, dim=0)
        sing_emb = torch.nn.functional.normalize(sing_emb, p=2, dim=0)
        
        return sing_emb, (self.mult_emb_label[index], self.data_label[index])
    
    
    # 여러개씩 sampling 되는 경우 사용
    def multiple_getitem(self, indices):
        
        sing_feat = []
        mult_feat = []
        
        for index in indices:
            
            sing_emb = numpy.load(self.sing_emb_list[index])
            mult_emb = self.mult_emb_label[index]
            
            # frame by 평균, 후 L2 Normalization
            sing_emb = torch.FloatTensor(sing_emb)
            sing_emb = torch.mean(sing_emb, dim=0)
            sing_emb = torch.nn.functional.normalize(sing_emb, p=2, dim=0)
            
            
            sing_emb = torch.unsqueeze(sing_emb, dim=0)
            mult_emb = torch.unsqueeze(mult_emb, dim=0)
            
            mult_feat.append(mult_emb)
            sing_feat.append(sing_emb)
        
        sing_feat = torch.cat(sing_feat, dim=0)
        mult_feat = torch.cat(mult_feat, dim=0)
        
        return sing_feat, (torch.FloatTensor(mult_feat), self.data_label[index])

    def __len__(self):
        return len(self.sing_emb_list)


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
        
        return embed, self.test_list[index]

    def __len__(self):
        return len(self.test_list)




class OriginalDataset(Dataset):
    
    def __init__(self, test_list, test_path, **kwargs):
        
        self.test_path  = test_path
        self.test_list = [i[:-3]+'npy' for i in test_list]

    def __getitem__(self, index):
        embed = numpy.load(os.path.join(self.test_path,self.test_list[index]))
        embed = torch.FloatTensor(embed)
        return embed, self.test_list[index]

    def __len__(self):
        return len(self.test_list)
    



'''
nPerSpeaker를 반영하는 데이터셋 sampler
각 batch에 다른 label(화자)의 데이터들이 들어가야 함.
최소의 개수를 가지는 label에 맞게 모든 label의 데이터 개수가 맞춰짐
train도 잘 안떨어지고, validation의 경우 크게 문제가 되는 것 같음
'''
class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.epoch              = 0
        self.seed               = seed
        self.distributed        = distributed
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        label2idx = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in label2idx):
                label2idx[speaker_label] = []
            label2idx[speaker_label].append(index)


        ## Group file indices for each class
        dictkeys = list(label2idx.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = [] # 같은 라벨을 가진 index들을 nPerSpeaker만큼씩 묶은 것들의 리스트
        flattened_label = [] # flattened_list에 순서에 맞는 라벨 (nPerSpeaker개의 발화에 맞는 라벨)
        
        for findex, key in enumerate(dictkeys):
            data    = label2idx[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            rp      = lol(numpy.arange(numSeg),self.nPerSpeaker)
            
            flattened_label.extend([findex] * (len(rp)))
            
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size) # 200, 400, 600, ...
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)
                
        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
