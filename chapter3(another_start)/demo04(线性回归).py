import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
# 定义学习率和次数等参数
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Training Data
# 训练和测试数据
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# 获取第0维长度
n_samples = train_X.shape[0]

# tf Graph Input
# 定义占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 定义生成模型的W和b（使用numpy随机数）
W = tf.Variable(rng.rand(), name="weight")
b = tf.Variable(rng.rand(), name="bias")

# 初始化线性回归方程(activation为X对应的Y值)
activation = tf.add(tf.multiply(X,W), b)

# 定义损失函数并用梯度下降算法降低损失函数
cost = tf.reduce_sum(tf.pow(activation-Y,2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

#  运行图
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            print("Y=",sess.run(W),"X+",sess.run(b))
            sess.run(optimizer,feed_dict={X:x,Y:y})

        if epoch % display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=",
                  "{:.9f}".format(sess.run(cost,feed_dict={X:train_X,Y:train_Y})),
                  "W=",sess.run(W),"b=",sess.run(b))

    print("Optimization Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),
          "W=",sess.run(W),"b=",sess.run(b))

    # 绘制图像
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()