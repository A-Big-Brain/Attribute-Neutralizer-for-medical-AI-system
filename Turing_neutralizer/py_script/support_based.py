import os
import random
import string
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics


path = '../../'
da_pa = path + 'test_data/'
mo_pa = path + 'Turing_modifier/'


def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''

    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1:]


# calculate the metrics
def cal_met(pred, lab):
    p_lab = pred.copy()
    p_lab[p_lab > 0.5] = 1
    p_lab[p_lab <= 0.5] = 0
    dif = 2*lab - p_lab
    met_li = []
    for i in range(p_lab.shape[-1]):
        li = dif[:, i]
        fn, tp, tn, fp = len(np.where(li == 2)[0]), len(np.where(li == 1)[0]), len(np.where(li == 0)[0]), len(np.where(li == -1)[0])
        acc = (tp + tn)/(fn + tp + tn + fp)
        sen = tp/(tp + fn + 1e-5)
        spe = tn/(tn + fp + 1e-5)
        met_li.append(np.array([acc, sen, spe]))
    met_li = np.stack(met_li, 0)
    return np.mean(met_li, 0)


# calculate the unfairness
def cl_met(g, pl):
    # performance
    fpr, tpr, thresholds = metrics.roc_curve(g, pl, pos_label=1)
    Auc = metrics.auc(fpr, tpr)
    pl[pl >= 0.5] = 1
    pl[pl < 0.5] = 0
    cm = confusion_matrix(g, pl)
    if cm.shape[0] < 2:
        return [-1, -1, -1, -1]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / (fn + tp + tn + fp)
    sen = tp / (tp + fn + 1e-4)
    spe = tn / (tn + fp + 1e-4)
    # unfairness
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    fdr = fp / (fp + tp)
    for1 = fn / (fn + tn)
    return np.array([acc, sen, spe, Auc]), np.array([fnr, fpr, fdr, for1])

# convert
def conv_one_hot(img, cal_num):
    mat_li = []
    for i in range(int(cal_num)):
        ma = img.copy()
        ma[np.where(ma == i)] = 100
        ma[np.where(ma != 100)] = 0
        ma[np.where(ma == 100)] = 1
        mat_li.append(ma)

    return np.stack(mat_li, -1)

# read args
def read_args(sa_fo, args):
    with open(sa_fo + 'args.txt', "w") as text_file:
        text_file.write(str(args))
    return