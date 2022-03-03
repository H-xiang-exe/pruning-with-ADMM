from model import LeNet5
from utils import loss_pretrain, loss_train, Initialize_Z_and_U, Update_Z, Update_U, apply_prune
from optimizer import PruneAdam
from torchvision.models.resnet import resnet50
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

'''--------------------------------------函数区---------------------------------------'''


def pretrain(args, model, train_loader, optimizer, test_loader, device):
    # 训练模型
    loss_list = []  # 存储每300个batch的平均损失
    for epoch in range(args.num_pre_epochs):
        print("Pre_Epoch:{}".format(epoch + 1))
        model.train()
        # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
        # model.train()是保证BN层能够用到每一批数据的均值和方差。
        # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零，初始化
            outputs = model(images)  # 前向传播
            loss = loss_pretrain(args, model, outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 权重更新

            running_loss += loss.item()  # 误差累积
            if batch_idx % 300 == 299:
                print('\npre_epoch:{} batch_idx:{} loss:{}'
                      .format(epoch + 1, batch_idx + 1, running_loss / 300))
                loss_list.append(running_loss / 300)
                running_loss = 0.0

    print("Finished PreTraining.")
    test(args, model, test_loader, device)

    return loss_list


def train(args, model, train_loader, optimizer, test_loader, device):
    # 初始化Z和U
    Z, U = Initialize_Z_and_U(model)
    # 开始训练，在约束下更新W,b
    running_loss = 0.0
    loss_list = []
    for epoch in range(args.num_epochs):
        print("Epoch:{}".format(epoch + 1))
        model.train()  # 使batch_norm和dropout能够生效
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            # 每一个batch更新一次w,b
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_train(args, model, outputs, labels, Z, U, device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 误差累积
            if batch_idx % 300 == 299:
                print('\nEpoch:{} batch_idx:{} loss:{}'
                      .format(epoch + 1, batch_idx + 1, running_loss / 300))
                loss_list.append(running_loss / 300)
                running_loss = 0.0
        # 每一个epoch更新一次Z和U
        Z = Update_Z(args, model, U)
        U = Update_U(model, Z, U)

    print("Finished Training.")
    test(args, model, test_loader, device)

    return loss_list


def retrain(args, model, train_loader, optimizer, mask, test_loader, device):
    loss_list = []  # 存储每300个batch的平均损失
    for epoch in range(args.num_re_epochs):
        print("Re_Epoch:{}".format(epoch + 1))
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零，初始化
            outputs = model(images)  # 前向传播
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)
            loss.backward()  # 反向传播
            optimizer.prune_step(mask)  # 权重更新

            running_loss += loss.item()  # 误差累积
            if batch_idx % 300 == 299:
                print('\npre_epoch:{} batch_idx:{} loss:{}'
                      .format(epoch + 1, batch_idx + 1, running_loss / 300))
                loss_list.append(running_loss / 300)
                running_loss = 0.0

    print("Finished ReTraining.")
    test(args, model, test_loader, device)

    return loss_list


def test(args, model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不进行计算图的构建，不跟踪梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += F.nll_loss(F.log_softmax(outputs,
                                    dim=1), labels, reduction='sum')
            pred = outputs.argmax(dim=1, keepdim=True)  # keepdim这个参数意味着什么
            correct += pred.eq(labels.view_as(pred)
                               ).sum().item()  # 通过item()从张量中获取元素

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%".format(test_loss,
                                                                       correct * 100.0 / len(test_loader.dataset)))


def calc_nums_zero_weight(model):
    num_zero_weight = ()
    for name, param in model.named_parameters():
        if name.split('.')[1] == 'weight':
            num_zero_weight += (torch.sum(param == 0).item(),)
    print(num_zero_weight)


'''--------------------------------------设定命令行参数阶段---------------------------------------'''
parser = argparse.ArgumentParser(description='training/testing settings')

parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10"],
                    help='training dataset (mnist or cifar10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int,
                    default=1024, help='input batch size for testing')
parser.add_argument('--num_pre_epochs', type=int, default=30)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--num_re_epochs', type=int, default=5)
parser.add_argument('--pretrain_lr', type=float, default=1e-2)
parser.add_argument('--train_lr', type=float, default=1e-3)
parser.add_argument('--retrain_lr', type=float, default=1e-2)
parser.add_argument('--adam_epsilon', type=float,
                    default=1e-8, help='adam epsilon (default: 1e-8)')
parser.add_argument('--alpha', type=float, default=1e-3,
                    help='l2 norm weight coefficient')
parser.add_argument('--rho', type=float, default=5e-5,
                    help='l2 penalty of W,Z')
parser.add_argument('--percent', type=list,
                    default=[80, 92, 99.1, 93], help='weight pruning ratio of per layer')

args = parser.parse_args()

'''--------------------------------------准备数据集阶段------------------------------------------'''

# 加载训练集
if (args.dataset == 'mnist'):
    training_data = datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
        download=True
    )

    train_loader = DataLoader(
        dataset=training_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 加载测试集
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.test_batch_size,
        shuffle=True
    )
elif (args.dataset == 'cifar10'):
    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                 (0.24703233, 0.24348505, 0.26158768))
        ]),
        download=True
    )

    train_loader = DataLoader(
        dataset=training_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 加载测试集
    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                     (0.24703233, 0.24348505, 0.26158768))
            ]
        ),
        download=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.test_batch_size,
        shuffle=True
    )

    args.percent = []
'''--------------------------------------加载模型阶段------------------------------------------'''
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 加载模型
if args.dataset == 'mnist':
    model = LeNet5().to(device)
else:
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model = model.to(device)
    args.percent = [
        80, 92, 99.1, 93, 93, 93, 93, 93, 93, 93,
        93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
        93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
        93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
        93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
    ]

# for name, param in model.named_parameters():
#     print(name)

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
pre_optimizer = PruneAdam(model.named_parameters(),
                          lr=args.pretrain_lr, eps=args.adam_epsilon)
optimizer = PruneAdam(model.named_parameters(),
                      lr=args.train_lr, eps=args.adam_epsilon)
re_optimizer = PruneAdam(model.named_parameters(),
                      lr=args.retrain_lr, eps=args.adam_epsilon)


'''--------------------------------------训练阶段------------------------------------------'''
loss_list = []
loss_val = pretrain(args, model, train_loader, pre_optimizer, test_loader, device)
loss_list.extend(loss_val)
calc_nums_zero_weight(model)

loss_val = train(args, model, train_loader, optimizer, test_loader, device)
loss_list.extend(loss_val)
calc_nums_zero_weight(model)

# 将接近于0的权值置0，剪枝
mask = apply_prune(args, model, device)
calc_nums_zero_weight(model)
test(args, model, test_loader, device)

loss_val = retrain(args, model, train_loader,
                  re_optimizer, mask, test_loader, device)
loss_list.extend(loss_val)
calc_nums_zero_weight(model)


# 打印损失变化值

plt.plot(loss_list)
plt.title('training loss')
plt.xlabel('epochs')
plt.ylabel('lose')
plt.savefig("loss.jpg")
plt.show()


torch.save(model, "./")
