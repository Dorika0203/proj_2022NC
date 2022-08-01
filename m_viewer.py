from matplotlib import pyplot as plt
import os


# SAVE_PATH = "exp_emb/temp"
# SAVE_PATH = "exp_emb/MyLinearNetv2_MSE_B200_E100"
SAVE_PATH = "exp_emb/MyLinearNetv2_Exp4"

file_path = os.path.join(SAVE_PATH, 'result/scores.txt')

f = open(file_path, 'r')


tloss = []
tepoch = []
vloss = []
vepoch = []
veer = []

for line in f.readlines():
    line = line.replace(',', '')
    tokens = line.strip().split()
    
    # train loss result
    if tokens[0].find('Val') == -1:
        tepoch.append(tokens[1])
        tloss.append(float(tokens[3]))
    
    else:
        vepoch.append(tokens[2])
        vloss.append(float(tokens[4]))
        veer.append(float(tokens[6]))

# plt.figure(figsize=(8, 8))
# plt.plot(tepoch, tloss, 'b-')
# plt.plot(vepoch, vloss, 'r-')

fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()
# ax1.plot(tepoch, tloss, **{'color': 'blue', 'marker': 'o'})
# ax1.plot(vepoch, vloss, **{'color': 'red', 'marker': 'o'})
# ax2.plot(vepoch, veer, **{'color': 'orange', 'marker': 'o'})
ax1.plot(tepoch, tloss, **{'color': 'blue'}, label='train loss')
ax1.plot(vepoch, vloss, **{'color': 'red'}, label='valid loss')
ax2.plot(vepoch, veer, **{'color': 'orange'}, label='valid EER')
fig.legend()
plt.savefig(os.path.join(SAVE_PATH, 'result/fig.png'))
f.close()
