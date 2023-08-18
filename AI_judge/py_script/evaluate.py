import torch
from torch import nn
import torchvision.models as models
import numpy as np
import support_based as spb
import support_dataset as spd
import support_net as spn
import os
import pickle
import datetime
import support_args as spa

# configuration
args = spa.parse()
sa_str = ['ChestX-ray14_age_4_120_100_3000_1000000_JscL', 'ChestX-ray14_gender_4_120_100_3000_1000000_0DIT'][0]
da_na = sa_str.split('_')[0]

# read data
if 'age' in sa_str:
    tar_fo = 'ChestX-ray14_age_64_500_100.0_10.0_0.0_QUhH'
else:
    tar_fo = 'ChestX-ray14_gender_64_500_100.0_10.0_0.0_qkC5'

or_img = np.load(spb.da_pa + da_na + '_img.npy', allow_pickle=True)
tra_img_li = [or_img]
for i in range(1, 2):
    img_na = spb.path + 'Data/Xray_generated_np/model26_generated/' + tar_fo + '/ChestX-ray14_' + str(10*i) + '.npy'
    if os.path.isfile(img_na):
        print(img_na)
        tar_img = np.load(img_na, allow_pickle=True)
        tra_img_li.append(tar_img)

info = np.load(spb.da_pa + da_na + '_info.npy', allow_pickle=True)
div = np.load(spb.da_pa + da_na + '_div_updated.npy', allow_pickle=True)

# convert to age and gender labels
gend_ind, age_ind = 3, 4
f_s, m_s = 'F', 'M'

# gender
# target label
if 'age' in sa_str:
    age_arr = np.zeros(info.shape[0])
    age_arr[info[:, age_ind] < 60] = 0
    age_arr[info[:, age_ind] >= 60] = 1
    age_arr = spb.conv_one_hot(age_arr, 2)
    tar_lab = age_arr
else:
    gend_arr = np.zeros(info.shape[0])
    gend_arr[info[:, gend_ind] == f_s] = 0
    gend_arr[info[:, gend_ind] == m_s] = 1
    gend_arr = spb.conv_one_hot(gend_arr, 2)
    tar_lab = gend_arr

# chose test data
te_ind = np.where(div == 2)[0]
tra_img_li = [x[te_ind] for x in tra_img_li]
tar_lab = tar_lab[te_ind]

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# construct the network
model = spn.get_model(2, 4)
model.load_state_dict(torch.load(spb.mo_pa + 'save_results/' + sa_str + '/' + 'model_weights.pth'))
model.to(device)
transforms = spn.get_transforms()
transforms.to(device)

# prediction
tar_pred_li = []

model.eval()
with torch.no_grad():
    for tra_img in tra_img_li:
        tar_pred = []
        for j in range(100):
            if j * args.bat_num > tra_img.shape[0] - 1:
                break

            bat_img = tra_img[j * args.bat_num: (j + 1) * args.bat_num, np.newaxis, :, :]
            bat_img = np.concatenate([bat_img, bat_img, bat_img], 1)
            bat_img = torch.tensor(bat_img / 255, dtype=torch.float32)
            bat_img = bat_img.to(device)

            bat_img = transforms(bat_img)
            log = model(bat_img)
            tar_pred.append(log.cpu())

            print(j)

        tar_pred = torch.concat(tar_pred).detach().numpy()
        tar_pred_li.append(tar_pred)

tar_pred_li = np.stack(tar_pred_li, 0)

# save
fi_na = spb.mo_pa + 'save_results/' + sa_str + '/prd.txt'
with open(fi_na, 'wb') as f:
    pickle.dump([tar_pred_li, tar_lab], f)