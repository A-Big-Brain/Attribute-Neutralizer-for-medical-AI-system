import torch
from torch import nn
import torchvision.models as Mod
import torchvision.transforms as tt
import datetime
import numpy as np
import support_based as spb
from functools import partial

def get_model(cla_num, model_num):
    if model_num == 0:
        model = Mod.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(1024, cla_num)
    elif model_num == 1:
        model = Mod.densenet201(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(1920, cla_num)
    elif model_num == 2:
        model = Mod.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(2048, cla_num)
    elif model_num == 3:
        model = Mod.resnet101(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(2048, cla_num)
    elif model_num == 4:
        model = Mod.convnext_small(weights='IMAGENET1K_V1')
        model.classifier[2] = nn.Linear(768, cla_num)
    elif model_num == 5:
        model = Mod.regnet_x_32gf(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(2520, cla_num)
    elif model_num == 6:
        model = Mod.efficientnet_b7(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(2560, cla_num)

    return model

def get_transforms():
    transforms = torch.nn.Sequential(
        #tt.Resize(size=600),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomRotation(degrees=(-10, 10), expand=False),
        tt.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        tt.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.12), scale=(0.9, 0.99)),
        tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms

# train the model
def train(da, model, transforms, loss_fn, optimizer, device, it_num, pr_it):
    model.train()
    all_los = 0
    for i in range(it_num):
        X, y1, _, _, _, _ = da.get_bat_data('tr')
        X, y1, = X.to(device), y1.to(device)

        # Compute prediction error
        X = transforms(X)
        log = model(X)

        # loss
        loss = loss_fn(log, y1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record
        all_los += loss.item()

        if i % pr_it == 0:
            print('train', i, datetime.datetime.now(), loss.item(), da.tr_i)

    return all_los

# test the model
def test(da, model, transforms, loss_fn, ty_str, device, it_num, pr_it):
    model.eval()
    test_loss, pred_li, lab_li, id_li = 0, [], [], []
    with torch.no_grad():
        for i in range(it_num):
            X, y1, _, _, _, idx = da.get_bat_data(ty_str)
            X, y1 = X.to(device), y1.to(device)

            # loss
            X = transforms(X)
            log = model(X)
            loss = loss_fn(log, y1).item()
            test_loss += loss

            # prediction
            pred = torch.sigmoid(log)
            pred_li.append(pred.cpu())
            lab_li.append(y1.cpu())
            id_li.append(idx)

            # whether stop
            if da.va_i == 0 and da.te_i == 0:
                break

            if i % pr_it == 0:
                print('validation', i, datetime.datetime.now(), loss)

    pred_li = torch.concat(pred_li).detach().numpy()
    lab_li = np.concatenate(lab_li, 0)
    id_li = np.concatenate(id_li)
    met = spb.cal_met(pred_li, lab_li)
    return test_loss, pred_li, id_li, met

