from matplotlib import pyplot as plt
import os

# SAVE_PATH = "exp_emb/MyLinearNet_Exp3-2"
# SAVE_PATH = "exp_emb/MyLinearNetv2_Exp6"

# SAVE_PATH = "exp_emb/MyLinearNetv2_Exp5-1-1"
# SAVE_PATH = "exp_emb/MyLinearNetv2_Exp5-3-1"

SAVE_PATH = "exp_emb/MyLinearNet_Exp3-2"
# SAVE_PATH = "exp_emb/MyLinearNetv2_Exp4-2"

file_path = os.path.join(SAVE_PATH, 'result/scores.txt')

f = open(file_path, 'r')


tloss = []
tepoch = []
vloss = []
vepoch = []
veer = []
veer_libri = []
# vacc = []

for line in f.readlines():
    line = line.replace(',', '')
    tokens = line.strip().split()
    
    # train loss result
    if tokens[0].find('Val') == -1:
        # if int(tokens[1]) > 40: break
        # if int(tokens[1]) < 3: continue
        tepoch.append(int(tokens[1]))
        tloss.append(float(tokens[3]))
    
    else:
        # if int(tokens[2]) < 3: continue
        vepoch.append(int(tokens[2]))
        vloss.append(float(tokens[4]))
        veer.append(float(tokens[6]))
        veer_libri.append(float(tokens[8]))
        # vacc.append(float(tokens[10]))

# fig, ax1 = plt.subplots(figsize=(8,8))
# ax2 = ax1.twinx()
# ax1.plot(tepoch, tloss, **{'color': 'blue'}, label='train loss')
# ax1.plot(vepoch, vloss, **{'color': 'red'}, label='valid loss')
# ax2.plot(vepoch, veer, **{'color': 'orange'}, label='valid EER')
# ax2.plot(vepoch, veer_libri, **{'color': 'pink'}, label='valid EER (libri)')
# # ax1.plot(vepoch, vacc, **{'color': 'yellow'}, label='valid MMD Loss')

fig, ax2 = plt.subplots(figsize=(8,8))
ax2.set_ylim(2.0, 3.0)
ax2.plot(vepoch, veer, **{'color': 'blue'}, label='valid EER')
ax2.plot(vepoch, veer_libri, **{'color': 'red'}, label='valid EER (libri)')

fig.legend()
# plt.savefig(os.path.join(SAVE_PATH, 'result/fig.png'))
plt.savefig(os.path.join(SAVE_PATH, 'result/fig2.png'))
f.close()
