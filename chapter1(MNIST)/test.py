import tensorflow as tf

import struct
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取MNIST数据集
def load_mnist(path,kind='train'):
    # 从path路径中导入数据
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)

    # 以二进制形式打开并使用struct的unpack进行读取
    # struct可以使用字节流打包和解包文件
    with open(labels_path,'rb') as lbpath:
        # 读取前八个字节（4字节的magic number，4字节的item）
        magic, n  = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)

    return images, labels

if __name__ == '__main__':
    # 使用matplotlib图形化显示获取的数据
    # 读取数据
    X_train, y_train = load_mnist('../data','train')

    # 设置显示格式
    fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,)

    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()