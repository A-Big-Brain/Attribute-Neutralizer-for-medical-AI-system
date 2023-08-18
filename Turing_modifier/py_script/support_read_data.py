import numpy as np
import support_based as spb
import random
import os

# read ChestX-ray14 data
def read_da_byname(da_na):
    # read files
    img = np.load(spb.da_pa + da_na + '_img.npy', allow_pickle=True)
    info = np.load(spb.da_pa + da_na + '_info.npy', allow_pickle=True)

    print(img.shape[0])

    # generate protected attributes
    age_co = 60
    if da_na == 'CheXpert':
        # gender
        gend_arr = np.zeros(info.shape[0], dtype=np.int32)
        gend_arr[info[:, 3] == 'Female'] = 0
        gend_arr[info[:, 3] == 'Male'] = 1
        # age
        age_arr = info[:, 4].astype(np.int32)
        age_arr[age_arr < age_co] = 0
        age_arr[age_arr >= age_co] = 1

        return img, info, gend_arr, age_arr

    elif da_na == 'ChestX-ray14':
        # gender
        gend_arr = np.zeros(info.shape[0], dtype=np.int32)
        gend_arr[info[:, 3] == 'F'] = 0
        gend_arr[info[:, 3] == 'M'] = 1
        # age
        age_arr = info[:, 4].astype(np.int32)
        age_arr[age_arr < age_co] = 0
        age_arr[age_arr >= age_co] = 1

        return img, info, gend_arr, age_arr

    elif da_na == 'MIMIC':
        # gender
        gend_arr = np.zeros(info.shape[0], dtype=np.int32)
        gend_arr[info[:, 2] == 'F'] = 0
        gend_arr[info[:, 2] == 'M'] = 1
        # age
        age_arr = info[:, 3].astype(np.int32)
        age_arr[age_arr < age_co] = 0
        age_arr[age_arr >= age_co] = 1
        # race
        race_arr = np.ones(info.shape[0], dtype=np.int32) * -1
        race_arr[info[:, 14] == 'WHITE'] = 0
        race_arr[info[:, 14] == 'HISPANIC/LATINO'] = 1
        race_arr[info[:, 14] == 'BLACK/AFRICAN AMERICAN'] = 2
        race_arr[info[:, 14] == 'ASIAN'] = 3
        race_arr[info[:, 14] == 'AMERICAN INDIAN/ALASKA NATIVE'] = 4
        race_arr[info[:, 14] == 'OTHER'] = 5
        # medi
        medi_arr = np.ones(info.shape[0], dtype=np.int32) * -1
        medi_arr[info[:, 11] == 'Medicaid'] = 0
        medi_arr[info[:, 11] == 'Medicare'] = 1
        medi_arr[info[:, 11] == 'Other'] = 2

        return img, info, gend_arr, age_arr, race_arr, medi_arr

def read_data(da_ty, pro_attr):
    if da_ty == 'MIMIC':
        img, info, gend_arr, age_arr, race_arr, medi_arr = read_da_byname('MIMIC')
    elif da_ty == 'ChestX-ray14':
        img, info, gend_arr, age_arr = read_da_byname('ChestX-ray14')
        race_arr = np.ones(info.shape[0], dtype=np.int32) * -1
        medi_arr = np.ones(info.shape[0], dtype=np.int32) * -1
    elif da_ty == 'CheXpert':
        img, info, gend_arr, age_arr = read_da_byname('CheXpert')
        race_arr = np.ones(info.shape[0], dtype=np.int32) * -1
        medi_arr = np.ones(info.shape[0], dtype=np.int32) * -1

    # disorder
    iind = np.arange(img.shape[0])
    random.shuffle(iind)
    img, info, gend_arr, age_arr, race_arr, medi_arr = img[iind], info[iind], gend_arr[iind], age_arr[iind], race_arr[iind], medi_arr[iind]

    if pro_attr == 'gender':
        return img, list(info), gend_arr[:, np.newaxis]
    elif pro_attr == 'age':
        return img, list(info), age_arr[:, np.newaxis]
    elif pro_attr == 'race':
        ind = np.where(race_arr != -1)[0]
        return img[ind, :, :], list(info[ind, :]), race_arr[:, np.newaxis][ind, :]
    elif pro_attr == 'medic':
        ind = np.where(medi_arr != -1)[0]
        return img[ind, :, :], list(info[ind, :]), medi_arr[:, np.newaxis][ind, :]
    elif pro_attr == 'gender_age':
        return img, list(info), np.stack([gend_arr, age_arr], -1)
    elif pro_attr == 'gender_age_race':
        ind = np.where(race_arr != -1)[0]
        return img[ind, :, :], list(info[ind, :]), np.stack([gend_arr, age_arr, race_arr], -1)[ind, :]
    elif pro_attr == 'gender_age_race_medic':
        ind1 = np.where(race_arr != -1)[0]
        ind2 = np.where(medi_arr != -1)[0]
        ind = np.intersect1d(ind1, ind2)
        return img[ind, :, :], list(info[ind, :]), np.stack([gend_arr, age_arr, race_arr, medi_arr], -1)[ind, :]

# read data for generate
def read_data_for_generate(da_ty, pro_attr):
    if da_ty == 'MIMIC':
        img, info, gend_arr, age_arr, race_arr, medi_arr = read_da_byname('MIMIC')
        race_arr = spb.conv_one_hot(race_arr, 6)
        medi_arr = spb.conv_one_hot(medi_arr, 3)
    elif da_ty == 'ChestX-ray14':
        img, info, gend_arr, age_arr = read_da_byname('ChestX-ray14')
    elif da_ty == 'CheXpert':
        img, info, gend_arr, age_arr = read_da_byname('CheXpert')

    gend_arr = gend_arr[:, np.newaxis]
    age_arr = age_arr[:, np.newaxis]

    if pro_attr == 'gender':
        return [[img, info, gend_arr]]
    elif pro_attr == 'age':
        return [[img, info, age_arr]]
    elif pro_attr == 'gender_age':
        return [[img, info, np.concatenate([gend_arr, age_arr], -1)]]
    elif pro_attr == 'race':
        return [[img, info, race_arr]]
    elif pro_attr == 'medic':
        return [[img, info, medi_arr]]
    elif pro_attr == 'gender_age_race':
        return [[img, info, np.concatenate([gend_arr, age_arr, race_arr], -1)]]
    elif pro_attr == 'gender_age_race_medic':
        return [[img, info, np.concatenate([gend_arr, age_arr, race_arr, medi_arr], -1)]]
