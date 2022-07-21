import os
import numpy as np
import torch
import warnings
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")
import shutil

'''
생성된 발화당 임베딩 벡터 (npy)들을
화자 단위의 다 발화 임베딩 벡터 하나로 정리한다.
'''

EMBED_DIR = "/home/doyeolkim/vox_emb/test/"


def generate_total_npy(target_path):
    
    '''
    target_path 내에 있는 모든 임베딩 벡터 npy 파일을
    평균을 내서 저장.
    '''
    
    np_list = []
    
    for root, directories, filenames in os.walk(target_path):
        for filename in filenames:
            if str.endswith(filename, '.npy'):
                emb = np.load(root + '/' + filename)
                
                assert emb.shape == (10,512)
                
                
                emb = np.mean(emb, axis=0)
                temp = torch.tensor(emb)
                print(temp.shape)
                emb2 = torch.nn.functional.normalize(temp, p=2)
                print(emb2.shape)
                
                emb = emb / np.linalg.norm(emb)
                print(np.max(np.abs(emb-emb2)))
                return
                np_list.append(emb)
    
    return np.mean(np.array(np_list), axis=0)


spk_list = []

for i in os.listdir(EMBED_DIR):
    if os.path.isdir(EMBED_DIR+i):
        spk_list.append(i)

print('spk_length: ', len(spk_list))
retval = generate_total_npy(EMBED_DIR+spk_list[0])

# print(spk_list[0], len(retval))
# print(retval[0].shape, retval[1].shape)        