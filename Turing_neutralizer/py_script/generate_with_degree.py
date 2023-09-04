import os
import torch
import numpy as np
import support_based as spb
import support_args as spa
import support_read_data as spr
from support_attgan import AttGAN

args = spa.parse()
sa_str = os.listdir(spb.mo_pa + 'save_results/')[args.train_aga_num]

# path
sa_fo = spb.mo_pa + 'save_results/' + sa_str + '/'

# save path
img_fo = spb.da_pa + sa_str + '/'
if not os.path.isdir(img_fo):
    os.mkdir(img_fo)

# dataset
str_li = sa_str.split('_')
da_ty = str_li[0]
pro_attr = ''
for s in str_li[1:-6]:
    pro_attr += s + '_'
pro_attr = pro_attr[:-1]
da_li = spr.read_data_for_generate(da_ty, pro_attr)

# attribute length
attli = ['gender', 'age', 'race', 'medic']
nli = [1, 1, 6, 3]
att_n = 0
for i, at in enumerate(attli):
    if at in sa_str:
        att_n += nli[i]

# model configuration
args.lr_base = args.lr
args.n_attrs = att_n
args.betas = (args.beta1, args.beta2)
print(args)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# construct model
fi_li = os.listdir(sa_fo)
fi_li = [x for x in fi_li if 'model_weights' in x]
fi_li = [int(x.split('_')[0]) for x in fi_li]
max_n = np.max(fi_li)
mo_fi = str(max_n) + '_model_weights.pth'

# load parameters
attgan = AttGAN(args)
attgan.load(sa_fo + mo_fi)
attgan.G.to(device)
attgan.D.to(device)

# evaluate
attgan.eval()
with torch.no_grad():
    for i in range(len(da_li)):
        da, att_li = da_li[i][0], da_li[i][2]
        att_li = (att_li * 2 - 1)
        if len(da_li) == 1:
            dana = da_ty
        else:
            dana = ['MIMIC', 'ChestX-ray14', 'CheXpert'][i]
        gen_da = []
        for j in range(args.epochs * args.iter_num):
            if j*args.batch_size > da.shape[0] - 1:
                break

            bat_img = da[j*args.batch_size: (j+1)*args.batch_size, np.newaxis, :, :]
            bat_img = 2 * (bat_img / 257) - 0.999
            bat_img = torch.tensor(bat_img, dtype=torch.float32).to(device)

            bat_att = att_li[j * args.batch_size: (j + 1) * args.batch_size, :]
            bat_att = bat_att * (-2 * args.edite_degree + 1)
            bat_att = torch.tensor(bat_att, dtype=torch.float32).to(device)

            gen_img = attgan.G(bat_img, bat_att).cpu().numpy()
            gen_img = (gen_img + 0.999) * 257 / 2
            gen_da.append(gen_img[:, 0, :, :].astype(np.uint8))

        gen_da = np.concatenate(gen_da, 0)
        # save
        fi_na = img_fo + dana + '_' + str(int(args.edite_degree*100)) + '.npy'
        np.save(fi_na, gen_da)