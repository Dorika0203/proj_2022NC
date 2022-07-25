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

# EMBED_DIR = "/home/doyeolkim/vox_emb/test/"
EMBED_DIR = "/home/doyeolkim/vox_emb/train/"


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
                # emb = emb / np.linalg.norm(emb)
                emb = np.array(torch.nn.functional.normalize(torch.tensor(emb), p=2, dim=0))
                
                np_list.append(emb)
    
    embs = np.array(np_list)
    return np.mean(embs, axis=0)

spk_list = []
for i in os.listdir(EMBED_DIR):
    if os.path.isdir(EMBED_DIR+i):
        spk_list.append(i)

for i, spk in enumerate(spk_list):
    mul_emb = generate_total_npy(EMBED_DIR+spk)
    np.save(EMBED_DIR+'{}.npy'.format(spk), mul_emb)
    print('\rprocessing ({}/{})'.format(i+1, len(spk_list)), end='')

print()
