import os
import random
import string
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

path = '../../'
da_pa = path + 'test_data/'
mo_pa = path + 'AI_judge/'


def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''

    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1:]

# calculate the metrics
def cal_met(pred, lab):
    # auc
    auc_li = []
    for i in range(pred.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(lab[:, i], pred[:, i], pos_label=1)
        Auc = metrics.auc(fpr, tpr)
        auc_li.append(Auc)

    # acc, sen, spe, and best cutoff value
    acc_li, sen_li, spe_li, cutoff_li = [], [], [], []
    for i in range(pred.shape[1]):
        pl, gr = pred[:, i], lab[:, i]
        cv = np.unique(pl)
        b_acc, b_sen, b_spe, b_cut = 0, 0, 0, 0

        # traverse
        for v in cv:
            p_lab = pl.copy()
            p_lab[p_lab > v] = 1
            p_lab[p_lab <= v] = 0
            dif = 2 * gr - p_lab

            # calculate
            fn, tp, tn, fp = len(np.where(dif == 2)[0]), len(np.where(dif == 1)[0]), len(np.where(dif == 0)[0]), len(np.where(dif == -1)[0])
            acc = (tp + tn) / (fn + tp + tn + fp)
            sen = tp / (tp + fn + 1e-5)
            spe = tn / (tn + fp + 1e-5)

            # judge
            if sen + spe > b_sen + b_spe:
                b_acc, b_sen, b_spe, b_cut = acc, sen, spe, v

        acc_li.append(b_acc)
        sen_li.append(b_sen)
        spe_li.append(b_spe)
        cutoff_li.append(b_cut)

    return np.array([auc_li, acc_li, sen_li, spe_li, cutoff_li])

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

def check(res, kn):
    if len(res[1]) < kn:
        return False
    loss = [x[0] for x in res[1]]
    cn = 0
    for i in range(kn):
        if loss[-1]- loss[-1-i] > 0:
            cn += 1
    if cn == kn:
        return True
    else:
        return False

    return


def cal_met_without_opt(pred, lab):
    # auc
    auc_li = []
    for i in range(pred.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(lab[:, i], pred[:, i], pos_label=1)
        Auc = metrics.auc(fpr, tpr)
        auc_li.append(Auc)

    # acc, sen, spe, and best cutoff value
    acc_li, sen_li, spe_li = [], [], []
    for i in range(pred.shape[1]):
        pl, gr = pred[:, i], lab[:, i]
        p_lab = pl.copy()
        p_lab[p_lab > 0.5] = 1
        p_lab[p_lab <= 0.5] = 0
        dif = 2 * gr - p_lab

        # calculate
        fn, tp, tn, fp = len(np.where(dif == 2)[0]), len(np.where(dif == 1)[0]), len(np.where(dif == 0)[0]), len(np.where(dif == -1)[0])
        acc = (tp + tn) / (fn + tp + tn + fp)
        sen = tp / (tp + fn + 1e-5)
        spe = tn / (tn + fp + 1e-5)

        acc_li.append(acc)
        sen_li.append(sen)
        spe_li.append(spe)

    return np.stack([auc_li, acc_li, sen_li, spe_li], -1)