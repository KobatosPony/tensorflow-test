import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# 定义参数
mb_size = 32
X_dim = 784
z_dim = 64
h_dim = 128
lr = 1e-3
m = 5
lam = 1e-3
gamma = 0.5
k_curr = 0

# 获取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 返回生成结果figure
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    # 指定子图的摆放位置
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# 使用xavier初始化权重
def xavier_init(size):
    in_dim = size[0]
    # 设定生成权重的方差
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# 样本输入的占位符
X = tf.placeholder(tf.float32, shape=[None, X_dim])
# 样本生成的占位符(generator)
z = tf.placeholder(tf.float32, shape=[None, z_dim])
# k值的占位符(在BEGAN中保证神经网络不会一直去优化generator)
k = tf.placeholder(tf.float32)

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]

# 在0-1随机生成均匀分布的噪点(size为shape)
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def D(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    X_recon = tf.matmul(D_h1, D_W2) + D_b2
    return tf.reduce_mean(tf.reduce_sum((X - X_recon)**2, 1))


G_sample = G(z)

D_real = D(X)
D_fake = D(G_sample)

D_loss = D_real - k*D_fake
G_loss = D_fake

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_real_curr = sess.run(
        [D_solver, D_real],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), k: k_curr}
    )

    _, D_fake_curr = sess.run(
        [G_solver, D_fake],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    # 更新限制参数k
    k_curr = k_curr + lam * (gamma*D_real_curr - D_fake_curr)

    if it % 1000 == 0:
        measure = D_real_curr + np.abs(gamma*D_real_curr - D_fake_curr)

        print('Iter-{}; Convergence measure: {:.4}'
              .format(it, measure))

        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)