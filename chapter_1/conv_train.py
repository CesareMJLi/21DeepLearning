# coding: utf-8
# test for one training process in tensor flow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    print("WEIGHT VARIABLE IS")
    print(initial)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    print("INITIAL VARIABLE IS")
    print(initial)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # here the strides means the filter would move 1 unit according to the input-4-dimension

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    # the input x has an arbitary number of images and each is a 784 dimension array
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # a 4 dimension input: image_number/x_axis/y_axis/channel

    # 第一层卷积层
    print("--------------------------CONV 1--------------------------------")
    W_conv1 = weight_variable([5, 5, 1, 32])
    # a filter: filter_size/filter_size/num_input_channels/num_filters
    b_conv1 = bias_variable([32])
    # make a 32-dimension vector
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Generally speaking, in each convolution layer, if we have n filters in this layer
    # we will have n outputs

    # 第二层卷积层
    print("--------------------------CONV 2--------------------------------")
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层，输出为1024维的向量
    print("--------------------------FULL CONNECTED--------------------------------")
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    print("--------------------------DROPOUT--------------------------------")
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    print("--------------------------TRANSFORM--------------------------------")
    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print("START TRAINING")

    # 训练20000步
    for i in range(1):
        batch = mnist.train.next_batch(50)
        # 每100步报告一次在验证集上的准确度
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 训练结束后报告在测试集上的准确度
    # print("test accuracy %g" % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
