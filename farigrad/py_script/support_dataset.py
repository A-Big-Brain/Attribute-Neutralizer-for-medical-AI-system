import support_read_data as srd
import support_based as spb
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

# dataset class
class dataset():
    def __init__(self, da_tr, attr_str, bat_num, lab_ind):
        # read data
        self.tr_da, self.va_da, self.te_da, self.lab_na = srd.read_da(da_tr)
        self.bat_num = bat_num
        self.lab_ind = lab_ind
        self.attr_str = attr_str
        self.attr_ind = ['gender', 'age', 'race', 'medi'].index(self.attr_str)
        self.cn = self.tr_da[1].shape[1]

        self.y = self.tr_da[1][:, self.lab_ind].astype(np.int32)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        self.s = self.tr_da[2][self.attr_ind]
        ind = np.where(self.s != -1)
        self.s = self.s[ind].astype(np.int32)
        self.y = self.y[ind]

        # index
        self.tr_i, self.va_i, self.te_i = 0, 0, 0
        self.tr_index = np.arange(self.tr_da[0].shape[0], dtype=np.uint16)
        self.va_index = np.arange(self.va_da[0].shape[0], dtype=np.uint16)
        self.te_index = np.arange(self.te_da[0].shape[0], dtype=np.uint16)

    def get_bat_index(self, index_list, bat_num, ind):
        if bat_num >= index_list.shape[0]:
            bat_index = index_list
            ind = 0
        else:
            st_num = (ind*bat_num)%len(index_list)
            bat_index = index_list[st_num: st_num + bat_num]
            if bat_index.shape[0] < bat_num:
                ind = 0
            elif bat_index.shape[0] == bat_num and (ind + 1)*bat_num == len(index_list):
                ind = 0
            else:
                ind = ind + 1
        return bat_index, ind

    def get_bat_data(self, ty_str):
        if ty_str == 'tr':
            bi, self.tr_i = self.get_bat_index(self.tr_index, self.bat_num, self.tr_i)
            da = self.tr_da
        elif ty_str == 'va':
            bi, self.va_i = self.get_bat_index(self.va_index, self.bat_num, self.va_i)
            da = self.va_da
        elif ty_str == 'te':
            bi, self.te_i = self.get_bat_index(self.te_index, self.bat_num, self.te_i)
            da = self.te_da

        atin = da[2][self.attr_ind][bi]
        ind = np.where(atin != -1)
        bi = bi[ind]

        img = np.stack([da[0][bi] for x in range(3)], 1)
        img = torch.tensor(img/255, dtype=torch.float32)
        if self.lab_ind == -1:
            lab = da[1][bi]
        else:
            lab = da[1][bi][:, self.lab_ind]

        return img, torch.tensor(lab), torch.tensor(da[2][self.attr_ind][bi]), da[3][bi], da[4][bi]
