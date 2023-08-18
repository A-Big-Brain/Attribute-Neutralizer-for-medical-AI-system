import support_read_data as srd
import support_based as spb
import numpy as np
import torch


# dataset class
class dataset():
    def __init__(self, da_tr, cla_attr, bat_num):
        # read data
        self.tr_da, self.va_da, self.te_da, self.lab_na = srd.read_da(da_tr)
        self.bat_num = bat_num
        self.cla_attr = cla_attr
        if cla_attr == 'gender':
            self.cn = self.tr_da[2].shape[1]
        elif cla_attr == 'age':
            self.cn = self.tr_da[3].shape[1]

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
        img = np.stack([da[0][bi] for x in range(3)], 1)
        img = torch.tensor(img/255, dtype=torch.float32)

        if self.cla_attr == 'gender':
            lab = da[2][bi]
        elif self.cla_attr == 'age':
            lab = da[3][bi]

        return img, torch.tensor(lab), torch.tensor(da[2][bi]), torch.tensor(da[3][bi]), da[4][bi], da[5][bi]
