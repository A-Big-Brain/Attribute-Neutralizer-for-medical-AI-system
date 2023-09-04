import support_read_data as srd
import support_based as spb
import numpy as np
import torch


# dataset class
class dataset():
    def __init__(self, da_ty, pro_attr, bat_num, image_c=1):
        # read data
        self.img, self.info, self.attr = srd.read_data(da_ty, pro_attr)
        self.bat_num = bat_num
        self.image_c = image_c

        # convert attribute
        self.na_li, self.con_attr = [], []
        for i in range(self.attr.shape[1]):
            a = self.attr[:, i]
            if a.max() == 1:
                b = a[:, np.newaxis]
                na = [pro_attr.split('_')[i]]
            else:
                b = np.zeros((a.size, a.max() + 1))
                b[np.arange(a.size), a] = 1
                na = [pro_attr.split('_')[i] + '_' + str(x)  for x in range(a.max() + 1)]
            self.na_li += na
            self.con_attr.append(b)
        self.con_attr = np.concatenate(self.con_attr, -1)

        # index
        self.tr_i = 0
        self.tr_index = np.arange(self.img.shape[0], dtype=np.uint32)

    # get the index
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


    def get_bat_data(self):
        bi, self.tr_i = self.get_bat_index(self.tr_index, self.bat_num, self.tr_i)
        # img
        img = 2*(self.img[bi]/257) - 0.999
        if np.random.random() > 0.5:
            img = np.flip(img, axis=2)
        img = torch.tensor(img[:,np.newaxis,:,:].copy(), dtype=torch.float32)

        return img, torch.tensor(self.con_attr[bi], dtype=torch.float32), [self.info[x] for x in bi]
