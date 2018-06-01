# tensorflow 的hello world
import tensorflow as tf

if __name__ == '__main__':
    # 通过constant 在TensorFlow中创建常量
    hello = tf.constant("Hello World!")
    sess = tf.Session()
    print(sess.run(hello))
