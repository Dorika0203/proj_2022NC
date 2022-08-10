'''
Loss (MSE) -> train_list, valid_list와 동일한 방식 (id10292 id10292/ENIHEvg_VLM/00019.npy)
EER -> test_list와 동일한 방식 (1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav)
'''

import os
import random

EMBED_DIR = "/home/doyeolkim/vox_emb/train/" # target embedding directory
FILENAME = "trainset" # filename to use for list

# EMBED_DIR = "/home/doyeolkim/vox_emb/test/" # target embedding directory
# FILENAME = "testset" # filename to use for list

# EMBED_DIR = "/home/doyeolkim/libri_emb/valid2/" # target embedding directory
# FILENAME = "libriset" # filename to use for list











spk_list = []
for i in os.listdir(EMBED_DIR):
    if os.path.isdir(EMBED_DIR+i):
        spk_list.append(i)












# spk_dict = {}
# for idx, spk in enumerate(spk_list):
#     sing_spk_list = []
    
#     for root, directories, filenames in os.walk(os.path.join(EMBED_DIR, spk)):
#         for filename in filenames:
#             if str.endswith(filename, 'npy'):
#                 sing_spk_list.append(str.replace(os.path.join(root, filename), EMBED_DIR, ''))
    
#     spk_dict[spk] = sing_spk_list
#     print('\rGenerating dictionary ({}/{})'.format(idx+1, len(spk_list)), end='')
# print()
# all_spk = list(spk_dict.keys())













# # normal list
# f = open('data/'+FILENAME+'_normal_list.txt', 'w')
# for idx, spk in enumerate(all_spk):
#     for filename in spk_dict[spk]:
#         write_line = spk + ' ' + filename + '\n'
#         f.write(write_line)
#     print('\rGenerating normal list ({}/{})'.format(idx+1, len(all_spk)), end='')
# f.close()














# # test_list
# f = open('data/'+FILENAME+'_test_list.txt', 'w')
# for idx, spk in enumerate(all_spk):
    
#     other_spk_list = all_spk.copy()
#     other_spk_list.remove(spk)
#     other_spks = list(spk_dict[other_spk_list[0]])
#     for i in other_spk_list[1:]:
#         other_spks.extend(list(spk_dict[i]))
    
#     for filename in spk_dict[spk]:
#         same_list = list(spk_dict[spk])
#         same_list.remove(filename)
        
#         same = random.sample(same_list, 4)
#         diff = random.sample(other_spks, 4)
        
#         for i in range(4):
#             f.write('1' + ' ' + filename + ' ' + same[i] + "\n")
#             f.write('0' + ' ' + filename + ' ' + diff[i] + "\n")
    
#     print('\rGenerating test list ({}/{})'.format(idx+1, len(all_spk)), end='')
# f.close()













# # triplet list
# f = open('data/'+FILENAME+'_triplet_list.txt', 'w')
# for idx, spk in enumerate(all_spk):
#     except_spk = all_spk.copy()
#     except_spk.remove(spk)
    
#     for filename in spk_dict[spk]:
#         write_line = spk + ' ' + filename + ' ' + random.choice(except_spk) + '\n'
#         f.write(write_line)
    
#     print('\rGenerating triplet list ({}/{})'.format(idx+1, len(all_spk)), end='')
    
# f.close()
# print()











# Distribution Adaptation List

'''
딕셔너리 value가 list일 때
최대 길이 list와 최소 길이 list인 (key, value) 튜플을 각각 반환
다른 카테고리의 두개를 고를 때 최대한 많이 포함하려면 이렇게 계속 뽑아야 함
'''
def select_minmax(target_dict):
    if len(target_dict) < 2:
        return None, None
    sorted_list = sorted(target_dict.items(), key=lambda item: len(item[1]))
    return sorted_list[0], sorted_list[-1]

f = open('data/'+FILENAME+'_distribution_list.txt', 'w')

spk_cat_dict = {}
for idx, spk in enumerate(spk_list):
            
    for root, directories, filenames in os.walk(os.path.join(EMBED_DIR, spk)):
        for filename in filenames:
            if str.endswith(filename, 'npy'):
                cat = str.split(root, '/')[-1]
                cat_dict = spk_cat_dict.get(spk, {})
                cat_list = cat_dict.get(cat, [])
                cat_list.append(str.replace(os.path.join(root, filename), EMBED_DIR, ''))
                cat_dict[cat] = cat_list
                spk_cat_dict[spk] = cat_dict
    print('\rGenerating dictionary ({}/{})'.format(idx+1, len(spk_list)), end='')
print()

same_cat_tups = []
diff_cat_tups = []

for idx, spk in enumerate(spk_list):
    cat_dict = spk_cat_dict[spk]
    cat_list = list(cat_dict.keys())
    counter = 0
    
    # same category files
    for cat in cat_list:
        file_list = cat_dict[cat]
        for f1, f2 in zip(file_list[0::2], file_list[1::2]):
            same_cat_tups.append((f1, f2))
            counter += 1
    
    # diff category files
    for _ in range(counter):
        i1, i2 = select_minmax(cat_dict)
        if i1 is not None:            
            f1 = random.choice(i1[1])
            f2 = random.choice(i2[1])
            i1[1].remove(f1)
            i2[1].remove(f2)
            if len(i1[1]) == 0: cat_dict.pop(i1[0])
            if len(i2[1]) == 0: cat_dict.pop(i2[0])
            diff_cat_tups.append((f1, f2))


for i in same_cat_tups:
    f.write(str.split(i[0], '/')[0] + ' ' + i[0] + ' ' + i[1] + ' 1\n')
for i in diff_cat_tups:
    f.write(str.split(i[0], '/')[0] + ' ' + i[0] + ' ' + i[1] + ' 0\n')