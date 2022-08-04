'''
Loss (MSE) -> train_list, valid_list와 동일한 방식 (id10292 id10292/ENIHEvg_VLM/00019.npy)
EER -> test_list와 동일한 방식 (1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav)
'''

import os
import random

# EMBED_DIR = "/home/doyeolkim/libri_emb/valid2/" # target embedding directory
# FILENAME = "valid2" # filename to use for list

# EMBED_DIR = "/home/doyeolkim/vox_emb/train/" # target embedding directory
# FILENAME = "trainset" # filename to use for list

# EMBED_DIR = "/home/doyeolkim/vox_emb/test/" # target embedding directory
# FILENAME = "testset" # filename to use for list

EMBED_DIR = "/home/doyeolkim/libri_emb/valid2/" # target embedding directory
FILENAME = "libriset" # filename to use for list

spk_list = []
for i in os.listdir(EMBED_DIR):
    if os.path.isdir(EMBED_DIR+i):
        spk_list.append(i)




spk_dict = {}
for idx, spk in enumerate(spk_list):
    sing_spk_list = []
    
    for root, directories, filenames in os.walk(os.path.join(EMBED_DIR, spk)):
        for filename in filenames:
            if str.endswith(filename, 'npy'):
                sing_spk_list.append(str.replace(os.path.join(root, filename), EMBED_DIR, ''))
    
    spk_dict[spk] = sing_spk_list
    print('\rGenerating dictionary ({}/{})'.format(idx+1, len(spk_list)), end='')

all_spk = list(spk_dict.keys())






# valid_list
f = open('data/'+FILENAME+'_normal_list.txt', 'w')
for idx, spk in enumerate(all_spk):
    for filename in spk_dict[spk]:
        write_line = spk + ' ' + filename + '\n'
        f.write(write_line)
    print('\rGenerating normal list ({}/{})'.format(idx+1, len(all_spk)), end='')
f.close()





# test_list
f = open('data/'+FILENAME+'_test_list.txt', 'w')
for idx, spk in enumerate(all_spk):
    
    other_spk_list = all_spk.copy()
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
    
    print('\rGenerating test list ({}/{})'.format(idx+1, len(all_spk)), end='')
f.close()




# triplet list
f = open('data/'+FILENAME+'_triplet_list.txt', 'w')
for idx, spk in enumerate(all_spk):
    except_spk = all_spk.copy()
    except_spk.remove(spk)
    
    for filename in spk_dict[spk]:
        write_line = spk + ' ' + filename + ' ' + random.choice(except_spk) + '\n'
        f.write(write_line)
    
    print('\rGenerating triplet list ({}/{})'.format(idx+1, len(all_spk)), end='')
    
f.close()

print()