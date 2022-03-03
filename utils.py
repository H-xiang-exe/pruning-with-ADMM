import numpy as np
import torch
import torch.nn.functional as F


def loss_pretrain(args, model, output, target):
    '''loss fucntion in pretrain stage'''
    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
    # print(loss)
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            loss += args.alpha * param.norm()  # 加上权重正则项，norm()表示2-norm矩阵范数
    return loss


def loss_train(args, model, output, target, Z, U, device):
    '''loss function in train stage'''
    idx = 0
    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            loss += args.alpha * param.norm()
            z = Z[idx].to(device)
            u = U[idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            idx += 1
    return loss


def Initialize_Z_and_U(model):
    '''initialize Z and U before train, make Z=W,U=0'''
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            Z += (param.detach().cpu().clone(),)
            # 这里detach的作用是从计算图中将param分离出来新的tensor并赋给z,其不受梯度反馈影响
            # cpu()返回一个此对象在cpu内存的副本，是一个临时对象
            # clone()在内存中新开辟一个空间存储对象
            # 这里这么操作的原因是我希望z=w，但同时有希望z和w都能各自更新，因此detach()之后z和w仅仅是共享内存，z不受梯度影响，但是z更新还是
            # 会影响到w，所以要再次clone()，这里直接clone()也是不行的，直接clone()会继承梯度信息，使得z的更新会受w的梯度的影响
            U += (torch.zeros_like(param).cpu(),)
    return Z, U


def Update_Z(args, model, U):
    new_Z = ()
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            w = param.detach().cpu().clone()
            u = U[idx]
            z = w + u
            pcen = np.percentile(abs(w),
                                 args.percent[idx])  # 求w在percent%上的分位数,不指定轴的话则是所有数的分位数（从小到大排列）,分位数公式见google
            # 掩码：权重绝对值从小到大排列前percent%以下的标记为True,在以上的标记为False
            under_threshold = abs(w) < pcen
            z.data[under_threshold] = 0  # 将较小的权重置为0；这里可以使用z.detach()吗？
            new_Z += (z,)
            idx += 1
    return new_Z


def Update_U(model, Z, U):
    new_U = ()
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            w = param.detach().cpu().clone()
            u = U[idx]
            z = Z[idx]
            new_u = u + w - z
            new_U += (new_u,)
            idx += 1
    return new_U


def prune_weight(weight, percent):
    '''对某一层的网络权重剪枝'''
    weight_numpy = weight.detach().cpu().numpy()

    pcen = np.percentile(abs(weight_numpy), percent)
    under_threshold = abs(weight_numpy) < pcen
    above_threshold = abs(weight_numpy) >= pcen

    return above_threshold


def apply_prune(args, model, device):
    idx = 0
    dict_mask = {}  # 存储每层的权重掩码
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight' and ('conv' in name or 'fc' in name):
            # print("at weight" + name)
            before = torch.sum(param != 0).item()
            # print("before pruning #non zero parameters:" + str(before)+'\n')

            # print('pruning current weight\n')
            mask = torch.tensor(prune_weight(param, args.percent[idx])).to(device)
            param.data.mul_(mask)
            dict_mask[name] = mask

            after = torch.sum(param != 0).item()
            # print("after pruning #non zero parameters:" + str(after)+'\n')
            idx += 1
    return dict_mask
