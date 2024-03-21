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

met_li = []
for dn in range(3):
    for trn in range(2):

        ar.da_num = dn
        if trn == 0:
            transforms = spn.get_transforms()
        else:
            transforms = spn.get_test_transforms()
        transforms.to(device)

        # dataset
        da = spd.dataset(da_li[ar.da_num], ar.bat_num, -1)

        # model path
        fi_li = ['ChestX-ray14_40_100_3000_1000000_NPWx', 'CheXpert_40_100_3000_1000000_2y3D', 'APPA_MIMIC_40_100_3000_1000000_c7no']
        PATH = 'E:/Project12_unfairness/Model/model19_classification_SOTA/save_results/' + fi_li[ar.da_num] + '/model_weights.pth'

        # construct the network
        model = spn.get_model(da.lab_na.shape[0])
        model.to(device)
        model.load_state_dict(torch.load(PATH))
        model.eval()


        # loss function
        loss_fn = nn.BCEWithLogitsLoss()

        test_loss, pred_li, lab_li, id_li = 0, [], [], []
        with torch.no_grad():
            for i in range(ar.te_it):
                X, y1, _, _, _, idx = da.get_bat_data('te')
                X, y1 = X.to(device), y1.to(device)

                # loss
                X = transforms(X)
                log = model(X)
                loss = loss_fn(log, y1).item()
                test_loss += loss

                # prediction
                pred = torch.sigmoid(log)
                pred_li.append(pred.cpu())
                lab_li.append(y1.cpu())
                id_li.append(idx)

                # whether stop
                if da.va_i == 0 and da.te_i == 0:
                    break

                if i % ar.pr_it == 0:
                    print('validation', i, datetime.datetime.now(), loss)

        pred_li = torch.concat(pred_li).detach().numpy()
        lab_li = np.concatenate(lab_li, 0)
        id_li = np.concatenate(id_li)
        met = spb.cal_met(pred_li, lab_li)

        met_li.append([da_li[dn], trn, met])

aa  = [np.mean(x[2], -1) for x in met_li]