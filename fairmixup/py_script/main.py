import support_args as spa
import support_dataset as spd
from torch import optim
import pickle

from support_net import *
from utils import *
from support_model import *


# configuration
args = spa.parse()

# create datasets
da_li = ['ChestX-ray14', 'CheXpert', 'MIMIC']
who_da = spd.dataset(da_tr=da_li[args.conf_num], attri_str='whole', attri_nint=0, bat_num=args.bat_num, lab_ind=-1)
attri_0_da = spd.dataset(da_tr=da_li[args.conf_num], attri_str=args.attri_str, attri_nint=0, bat_num=args.bat_num, lab_ind=-1)
attri_1_da = spd.dataset(da_tr=da_li[args.conf_num], attri_str=args.attri_str, attri_nint=1, bat_num=args.bat_num, lab_ind=-1)

# create folder
sa_str = spb.com_mul_str([da_li[args.conf_num], args.attri_str, args.mode, args.lam, args.bat_num, args.epoch, args.tr_it, args.te_it])
sa_fo = spb.mo_pa + 'save_results/' + sa_str + '/'
os.mkdir(sa_fo)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model
model = convnext_Encoder().to(device)
model_linear = LinearModel(who_da.cn).to(device)

transforms = get_transforms().to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight = who_da.cal_w).to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.0001)
optimizer_linear = optim.SGD(model_linear.parameters(), lr = 0.0001)

# train
res = [[], [], []]
for i in range(1, args.epoch):

    if spb.check(res, 4):
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
        optimizer_linear.param_groups[0]['lr'] = optimizer_linear.param_groups[0]['lr'] / 2

    tr_loss_sup, tr_loss_gap = train(i, model, model_linear, transforms, criterion, optimizer, optimizer_linear, device, attri_0_da, attri_1_da, args)

    # validation
    va_loss, va_pred, va_id, va_met = test(model, model_linear, transforms, criterion, device, who_da, 'va', args)
    print('all_validation', i, va_loss, np.mean(va_met, 1))

    # test
    te_loss, te_pred, te_id, te_met = test(model, model_linear, transforms, criterion, device, who_da, 'te', args)
    print('all_test', i, te_loss, np.mean(te_met, 1))

    # add
    res[0].append([tr_loss_sup, tr_loss_gap])
    res[1].append([va_loss, va_pred, va_id, va_met])
    res[2].append([te_loss, te_pred, te_id, te_met])

    # save the result
    with open(sa_fo + 'res.txt', 'wb') as fi:
        pickle.dump(res, fi)

    # save the model
    torch.save(model.state_dict(), sa_fo + 'model_weights.pth')

    # whether stop:
    if spb.check(res, 100):
        break
