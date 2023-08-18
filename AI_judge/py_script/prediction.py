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

ar = spa.parse()
da_li = ['ChestX-ray14', 'CheXpert', 'APPA_MIMIC']
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

for dn in range(1):
    ar.da_num = dn
    # dataset
    da = spd.dataset(da_li[ar.da_num], ar.bat_num, -1)

    # model path
    fi_li = ['ChestX-ray14_40_100_3000_1000000_NPWx', 'CheXpert_40_100_3000_1000000_2y3D', 'APPA_MIMIC_40_100_3000_1000000_c7no']
    sa_fo = spb.mo_pa + 'save_results/' + fi_li[ar.da_num] + '/'
    PATH = sa_fo + 'model_weights.pth'

    # construct the network
    model = spn.get_model(da.lab_na.shape[0])
    model.to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    transforms = spn.get_transforms()
    transforms.to(device)

    pred_li, lab_li, id_li = [], [], []
    with torch.no_grad():
        for ty_str in ['tr', 'te', 'va']:
            for i in range(1000000):
                X, y1, _, _, _, idx = da.get_bat_data(ty_str)
                X, y1 = X.to(device), y1.to(device)

                # loss
                X = transforms(X)
                log = model(X)

                # prediction
                pred = torch.sigmoid(log)
                pred_li.append(pred.cpu())
                lab_li.append(y1.cpu())
                id_li.append(idx)

                # whether stop
                if da.va_i == 0 and da.te_i == 0 and da.tr_i == 0:
                    break

                print(ty_str, i, da.va_i, da.te_i, da.tr_i)

    pred_li = torch.concat(pred_li).detach().numpy()
    lab_li = np.concatenate(lab_li, 0)
    id_li = np.concatenate(id_li)

    with open(sa_fo + 'pred.txt', 'wb') as fi:
        pickle.dump([pred_li, lab_li, id_li], fi)

