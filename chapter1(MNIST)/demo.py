import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import struct
import numpy as np
import os

if __name__ == '__main__':
    # 导入数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 使用softmax回归模型进行计算（需要权重和偏置量）
    # 设置占位符
    x = tf.placeholder('float',[None,784])
    #设置初始的权重和偏置量(初始都设置为0)
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # 实现模型
    # matmul表示张量相乘(对应模型中使用矩阵相乘来计算权值)
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # 使用交叉熵来作为指标评估模型
    # 添加一个占位符
    y_ = tf.placeholder('float',[None,10])

    # 计算交叉熵(reduce_sum用来计算张量的所有元素的总和)
    # 这里交叉熵为所有100幅图片的交叉熵总和
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # tensorflow 会自动使用反向传播算法来确定你的变量
    # 是如何影响你想要的最小化的那个成本值

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 在这里，我们要求TensorFlow用梯度下降算法（gradient
    # descent algorithm）以0.01的学习速率最小化交叉熵。

    # 初始化创建的变量
    init = tf.global_variables_initializer()

    # 启动模型并初始化变量
    sess = tf.Session()
    sess.run(init)

    # 开始训练
    # 让模型循环训练一千次
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

    # 我们都会随机抓取训练数据中的100个批处理数据点，
    # 然后我们用这些数据点作为参数替换之前的占位符
    # 来运行train_step

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))