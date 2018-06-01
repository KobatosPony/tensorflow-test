from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    #加载 mnist 数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 使用更方便的 InteractiveSession 代替session可以让你在运行图的时候插入一些计算图
    sess = tf.InteractiveSession()

    # 定义占位符
    x = tf.placeholder("float",shape=[None,784])
    _y = tf.placeholder("float",shape=[None,10])

    # 定义权重和偏置值（这里都设为0）
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # 变量需要通过session初始化之后啊才能使用
    sess.run(tf.global_variables_initializer())

    # 构建回归模型
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # 计算batch中的每张图片的交叉熵之和
    cross_entropy = -tf.reduce_sum(_y * tf.log(y))

    # 使用梯度下降算法以0.01的速率让交叉熵下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行train_step来完成。

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x:batch[0],_y:batch[1]})

    # 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
    # 注意，在计算图中，你可以用feed_dict来替代任何张量，并不仅限于替换占位符

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(_y,1))

    # 这里会返回一个布尔数组
    # 使用如下方法取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

    print(accuracy.eval(feed_dict={x:mnist.test.images,_y:mnist.test.labels}))