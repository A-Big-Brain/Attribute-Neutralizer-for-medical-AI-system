import torch
import datetime
import numpy as np
import support_based as spb
from numpy.random import beta


def train(epochs, model, model_linear, transforms, criterion, optimizer, optimizer_linear, device, attri_0_da, attri_1_da, args):
    print("Epoch: {}".format(epochs))

    model.train()
    model_linear.train()
    all_loss_sup, all_loss_gap = 0, 0
    for it in range(args.tr_it):
        inputs_0, target_0, _, _, _, _ = attri_0_da.get_bat_data('tr')
        inputs_1, target_1, _, _, _, _ = attri_1_da.get_bat_data('tr')
        inputs_0, target_0 = inputs_0.to(device), target_0.float().to(device)
        inputs_1, target_1 = inputs_1.to(device), target_1.float().to(device)

        inputs_0 = transforms(inputs_0)
        inputs_1 = transforms(inputs_1)

        # check size
        if inputs_0.shape[0] != inputs_1.shape[0]:
            continue

        # supervised loss
        inputs = torch.cat((inputs_0, inputs_1), 0)
        target = torch.cat((target_0, target_1), 0)

        feat = model(inputs)
        ops = model_linear(feat)
        loss_sup = criterion(ops, target)

        if args.mode == 'GapReg':
            feat = model(inputs_0)
            ops_0 = model_linear(feat)
            feat = model(inputs_1)
            ops_1 = model_linear(feat)

            loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
            loss = loss_sup + args.lam*loss_gap

            all_loss_sup += loss_sup.item()
            all_loss_gap += loss_gap.item()

            if it % args.pr_it == 0:
                print("Loss Sup: {:.4f} | Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif args.mode == 'mixup':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Input Mixup
            inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            feat = model(inputs_mix)
            ops = model_linear(feat).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())

            loss = loss_sup + args.lam * loss_grad

            all_loss_sup += loss_sup.item()
            all_loss_gap += loss_grad.item()

            if it % args.pr_it == 0:
                print("Loss Sup: {:.4f} Loss Mixup {:.7f}".format(loss_sup, loss_grad))

        elif args.mode == 'mixupmanifold':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Manifold Mixup
            feat_0 = model(inputs_0)
            feat_1 = model(inputs_1)
            inputs_mix = feat_0 * gamma + feat_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            ops = model_linear(inputs_mix).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())

            loss = loss_sup + args.lam * loss_grad

            all_loss_sup += loss_sup.item()
            all_loss_gap += loss_grad.item()

            if it % args.pr_it == 0:
                print("Loss Sup: {:.4f} Loss Mixup Manifold {:.7f}".format(loss_sup, loss_grad))

        else:
            loss = loss_sup

            all_loss_sup += loss_sup.item()
            all_loss_gap += 0

            if it % args.pr_it == 0:
                print("Loss Sup: {:.4f}".format(loss_sup))

        optimizer.zero_grad()
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_linear.step()

    return all_loss_sup, all_loss_gap


def test(model, model_linear, transforms, criterion, device, who_da, ty_str, args):
    model.eval()
    model_linear.eval()
    test_loss, pred_li, lab_li, id_li = 0, [], [], []
    with torch.no_grad():
        for i in range(args.te_it):
            X, y1, _, _, _, idx = who_da.get_bat_data(ty_str)
            X, y1 = X.to(device), y1.to(device)

            # loss
            feat = model(X)
            log = model_linear(feat)

            loss = criterion(log, y1).item()
            test_loss += loss

            # prediction
            pred = torch.sigmoid(log)
            pred_li.append(pred.cpu())
            lab_li.append(y1.cpu())
            id_li.append(idx)

            # whether stop
            if who_da.va_i == 0 and who_da.te_i == 0:
                break

            if i % args.pr_it == 0:
                print('validation', i, datetime.datetime.now(), loss)

    pred_li = torch.concat(pred_li).detach().numpy()
    lab_li = np.concatenate(lab_li, 0)
    id_li = np.concatenate(id_li)
    met = spb.cal_met(pred_li, lab_li)
    return test_loss, pred_li, id_li, met


