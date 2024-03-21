import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import support_dataset as spd
import support_based as spb
import support_args as spa
from support_attgan import AttGAN

args = spa.parse()

# dataset
conf_li = [['MIMIC', x] for x in ['gender', 'age', 'gender_age', 'race', 'medic', 'gender_age_race', 'gender_age_race_medic']] + \
          [[x, y] for x in ['ChestX-ray14', 'CheXpert'] for y in ['gender', 'age', 'gender_age']]

[da_ty, pro_attr] = conf_li[args.conf_num]
da = spd.dataset(da_ty=da_ty, pro_attr=pro_attr, bat_num=args.batch_size, image_c=1)

# configuration
args.lr_base = args.lr
args.n_attrs = da.con_attr.shape[1]
args.betas = (args.beta1, args.beta2)
print(args)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# construct the model
attgan = AttGAN(args)
attgan.G.to(device)
attgan.D.to(device)

# create folder
sa_str = spb.com_mul_str([da_ty, pro_attr, args.batch_size, args.epochs, args.lambda_1, args.lambda_2, args.update_lambda_rate])
sa_fo = spb.mo_pa + 'save_results/' + sa_str + '/'
img_fo = sa_fo + 'images/'
os.mkdir(sa_fo)
os.mkdir(img_fo)
spb.read_args(sa_fo, args)

# train
tr_res, errD, errG = [args], 0, 0
te_img_a, te_att_a, _ = da.get_bat_data()
for epoch in range(args.epochs):
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    for i in range(args.iter_num):
        attgan.train()
        img_a, att_a, _ = da.get_bat_data()

        img_a = img_a.to(device)
        att_a = att_a.to(device)
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)

        #att_a_ = (att_a * 2 - 1) * args.thres_int
        if args.b_distribution == 'none':
            att_b_ = (att_b * 2 - 1) * args.thres_int
            att_a_ = (att_a * 2 - 1) * args.thres_int
        if args.b_distribution == 'uniform':
            att_b_ = (att_b * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
            att_a_ = (att_a * 2 - 1) * torch.rand_like(att_b) * (2 * args.thres_int)
        if args.b_distribution == 'truncated_normal':
            att_b_ = (att_b * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)
            att_a_ = (att_a * 2 - 1) * (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * (2 * args.thres_int)

        if (i + 1) % (args.n_d + 1) != 0:
            errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)
        else:
            errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)

        tr_res.append([errD, errG])

        # print
        if (i + 1) % 5 == 0:
            print(i, errD)
            print(i, errG)

        # generate images
        if (i + 1) % args.sample_interval == 0:

            # create folder
            te_im_fo = img_fo + str(epoch) + '_' + '0000'[:-len(str(i))] + str(i) + '_' + str(attgan.lambda_1) + '_' + str(attgan.lambda_2)[:5] + '/'
            os.mkdir(te_im_fo)

            # evaluate
            attgan.eval()
            with torch.no_grad():
                att_b_ = (te_att_a * 2 - 1)
                for m in range(te_att_a.shape[0]):
                    att_li = []
                    for k in range(11):
                        at = att_b_[m, :]
                        at = at*(0.2*k-1)
                        att_li.append(at)
                    img_li = torch.stack([te_img_a[m] for x in range(11)], 0)
                    att_li = torch.stack(att_li, 0)
                    img_li = img_li.to(device)
                    att_li = att_li.to(device)

                    pred_img = attgan.G(img_li, att_li)
                    pred_img = pred_img.cpu().numpy()
                    ima = te_img_a[m, 0, :, :].cpu().numpy()
                    pred_img = np.concatenate([ima[np.newaxis, np.newaxis, :,:], pred_img], 0)

                    pred_img = (pred_img + 0.999)*257/2
                    pred_img = [pred_img[x, 0,:,:] for x in range(pred_img.shape[0])]
                    pred_img = np.concatenate(pred_img, 1)

                    # save as images
                    att = att_b_[m, :].cpu().numpy()
                    fi_na = te_im_fo + str(m) + '_' + str(att) + '.jpg'
                    plt.imsave(fi_na, np.stack([pred_img for x in range(3)], -1).astype(np.uint8))

    # update lambda_2
    attgan.update_lambda()

    # save model and result
    attgan.save(sa_fo + str((epoch//25)*25) + '_model_weights.pth')
    with open(sa_fo + 'tr_res.txt', 'wb') as fi:
        pickle.dump(tr_res, fi)


