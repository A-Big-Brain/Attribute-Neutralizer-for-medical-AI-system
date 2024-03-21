import numpy as np
import support_based as spb
import random

# read ChestX-ray14 data
def read_da(da_tr, at_str, nint):
    # read files
    da_tr = da_tr.replace('MIMIC', 'APPA_MIMIC')
    img = np.load(spb.da_pa + da_tr + '_img.npy', allow_pickle=True)
    lab = np.load(spb.da_pa + da_tr + '_lab.npy', allow_pickle=True).astype(np.float32)
    lab = np.nan_to_num(lab)
    lab[lab == -1] = 0
    info = np.load(spb.da_pa + da_tr + '_info.npy', allow_pickle=True)
    div = np.load(spb.da_pa + da_tr + '_div_updated.npy', allow_pickle=True)
    lab_na = np.load(spb.da_pa + da_tr.replace('APPA_', '') + '_lab_na.npy', allow_pickle=True)

    # generate protected attributes
    if da_tr == 'CheXpert':
        gend_ind, age_ind = 3, 4
        f_s, m_s = 'Female', 'Male'
    elif da_tr == 'ChestX-ray14':
        gend_ind, age_ind = 3, 4
        f_s, m_s = 'F', 'M'
    elif da_tr == 'APPA_MIMIC':
        gend_ind, age_ind = 2, 3
        f_s, m_s = 'F', 'M'

    # gender
    gend_arr = np.zeros(info.shape[0])
    gend_arr[info[:, gend_ind] == f_s] = 0
    gend_arr[info[:, gend_ind] == m_s] = 1
    gend_ind = [np.where(gend_arr == x)[0] for x in range(2)]

    # age
    age_co = 60
    age_arr = info[:, age_ind].copy().astype(np.float64)
    age_arr[age_arr < age_co] = 0
    age_arr[age_arr >= age_co] = 1
    age_ind = [np.where(age_arr == x)[0] for x in range(2)]

    # divide training, test, and validation
    [tr_ind, va_ind, te_ind] = [np.where(div == x)[0] for x in range(3)]

    # the whole dataset
    if at_str != 'whole':
        if at_str == 'gender':
            attr_arr = gend_ind[nint]
        elif at_str == 'age':
            attr_arr = age_ind[nint]
        else:
            print('attribute string is wrong')
        tr_ind = np.intersect1d(tr_ind, attr_arr)
        va_ind = np.intersect1d(va_ind, attr_arr)
        te_ind = np.intersect1d(te_ind, attr_arr)

    random.shuffle(tr_ind)
    random.shuffle(va_ind)
    random.shuffle(te_ind)
    tr_da = [img[tr_ind], lab[tr_ind], gend_arr[tr_ind], age_arr[tr_ind], info[tr_ind], tr_ind]
    va_da = [img[va_ind], lab[va_ind], gend_arr[va_ind], age_arr[va_ind], info[va_ind], va_ind]
    te_da = [img[te_ind], lab[te_ind], gend_arr[te_ind], age_arr[te_ind], info[te_ind], te_ind]


    return tr_da, va_da, te_da, lab_na
