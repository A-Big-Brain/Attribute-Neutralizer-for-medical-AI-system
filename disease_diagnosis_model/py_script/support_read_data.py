import numpy as np
import support_based as spb
import random

# read ChestX-ray14 data
def read_da(da_tr, folder):
    # read files
    img = np.load(spb.da_pa + folder + '/' + da_tr + '_50.npy', allow_pickle=True)
    lab = np.load(spb.da_pa + da_tr + '_lab.npy', allow_pickle=True).astype(np.float32)
    lab = np.nan_to_num(lab)
    lab[lab == -1] = 0
    info = np.load(spb.da_pa + da_tr + '_info.npy', allow_pickle=True)
    div = np.load(spb.da_pa + da_tr + '_div.npy', allow_pickle=True)
    lab_na = np.load(spb.da_pa + da_tr+ '_lab_na.npy', allow_pickle=True)

    # divide
    [tr_ind, va_ind, te_ind] = [np.where(div == x)[0] for x in range(3)]
    random.shuffle(tr_ind)
    random.shuffle(va_ind)
    random.shuffle(te_ind)
    tr_da = [img[tr_ind], lab[tr_ind], info[tr_ind], tr_ind]
    va_da = [img[va_ind], lab[va_ind], info[va_ind], va_ind]
    te_da = [img[te_ind], lab[te_ind], info[te_ind], te_ind]

    return tr_da, va_da, te_da, lab_na