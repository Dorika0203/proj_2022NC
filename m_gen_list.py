'''
Loss (MSE) -> train_list, valid_list와 동일한 방식 (id10292 id10292/ENIHEvg_VLM/00019.npy)
EER -> test_list와 동일한 방식 (1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav)
'''

import os
import random

EMBED_DIR = "/home/doyeolkim/libri_emb/valid2/" # target embedding directory
FILENAME = "valid2" # filename to use for list

spk_list = []
for i in os.listdir(EMBED_DIR):
    if os.path.isdir(EMBED_DIR+i):
        spk_list.append(i)

spk_dict = {}
for i, spk in enumerate(spk_list):
    sing_spk_list = []
    
    for root, directories, filenames in os.walk(os.path.join(EMBED_DIR, spk)):
        for filename in filenames:
            if str.endswith(filename, 'npy'):
                sing_spk_list.append(str.replace(os.path.join(root, filename), EMBED_DIR, ''))
    
    spk_dict[spk] = sing_spk_list






# valid_list
f = open('data/'+FILENAME+'_valid_list.txt', 'w')
for spk in list(spk_dict.keys()):
    for filename in spk_dict[spk]:
        write_line = spk + ' ' + filename + '\n'
        f.write(write_line)

f.close()




# test_list
f = open('data/'+FILENAME+'_test_list.txt', 'w')
for spk in list(spk_dict.keys()):
    
    other_spk_list = list(spk_dict.keys())
    other_spk_list.remove(spk)
    other_spks = list(spk_dict[other_spk_list[0]])
    for i in other_spk_list[1:]:
        other_spks.extend(list(spk_dict[i]))
    
    for filename in spk_dict[spk]:
        same_list = list(spk_dict[spk])
        same_list.remove(filename)
        
        same = random.sample(same_list, 4)
        diff = random.sample(other_spks, 4)
        
        for i in range(4):
            f.write('1' + ' ' + filename + ' ' + same[i] + "\n")
            f.write('0' + ' ' + filename + ' ' + diff[i] + "\n")
f.close()
        