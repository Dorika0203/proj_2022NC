from matplotlib import pyplot as plt
import os


# SAVE_PATH = "exp_emb/temp"
# SAVE_PATH = "exp_emb/MyLinearNetv2_MSE_B200_E100"
SAVE_PATH = "exp_emb/MyLinearNetv2_MSE_B100_E100"

file_path = os.path.join(SAVE_PATH, 'result/scores.txt')

f = open(file_path, 'r')


tloss = []
tepoch = []
vloss = []
vepoch = []

for line in f.readlines():
    tokens = line.strip().split()
    
    # train loss result
    if tokens[0].find('Val') == -1:
        tepoch.append(tokens[1])
        tloss.append(float(tokens[3][:-1]))
    
    else:
        vepoch.append(tokens[2])
        vloss.append(float(tokens[4][:-1]))


plt.figure(figsize=(8, 8))

# all_loss = tloss.extend(vloss)
# plt.ylim((min(all_loss), max(all_loss)))
plt.plot(tepoch, tloss, 'b-')
plt.plot(vepoch, vloss, 'r-')
plt.savefig(os.path.join(SAVE_PATH, 'result/fig.png'))
f.close()