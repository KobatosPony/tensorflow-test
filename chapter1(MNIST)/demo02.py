from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # 加载 mnist 数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 使用更方便的 InteractiveSession 代替session可以让你在运行图的时候插入一些计算图
    sess = tf.InteractiveSession()
