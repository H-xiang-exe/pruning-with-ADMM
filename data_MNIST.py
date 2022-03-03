import os
import numpy as np
from torch.utils.data import Dataset
import gzip


class Mnist(Dataset):
    def __init__(self, root, train=True, transform=None):
        # root是数据集存放目录，train=True表示词数据集是训练集

        # 根据是否为训练集，得到文件名前缀
        self.file_pre = 'train' if train == True else 't10k'
        self.transform = transform

        # 生成对应数据集的图片和标签文件的路径
        self.label_path = os.path.join(root, '%s-labels-idx1-ubyte.gz' % self.file_pre)
        self.image_path = os.path.join(root, '%s-images-idx3-ubyte.gz' % self.file_pre)

        # 读取文件数据，返回图片和标签
        self.images, self.labels = self.__read_data__(self.image_path, self.label_path)

    '''读取数据集'''

    def __read_data__(self, image_path, label_path):
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)
        return images, labels

    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])

        # 如果需要转成tensor 则使用transform
        if self.transform is not None:
            image = self.transform(np.array(image))  # 此处需要用np.array
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train_set = Mnist(root=r'D:\LearningWorks\PythonWorks\admm-pruning\mnist_dataset',
                      train=True)

    (data, label) = train_set[1]
    print(label)
    import matplotlib.pyplot as plt
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.title('label is: {}'.format(label))
    plt.show()