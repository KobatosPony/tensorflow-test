# 实现简单的数学方法
import tensorflow as tf

if __name__ == '__main__':
    # 定义常量
    a = tf.constant(2)
    b = tf.constant(3)

    # 使用Session的run来启动整个运算图
    with tf.Session() as sess:
        print("a=2, b=3")
        print("Addition with constants: %i" % sess.run(a+b))
        print("Multiplication with constants: %i" % sess.run(a*b))
