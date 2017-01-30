import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE_X = 28
IMAGE_SIZE_Y = 28
NUM_IMAGE_CHANNELS = 1
NUM_CLASS = 10

flags = tf.app.flags
DEFINE = flags.FLAGS

flags.DEFINE_float('random_normal_stddev', 5e-2, 'Random normal standard deviation')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')

flags.DEFINE_integer('epochs', 10000, 'Number of epochs')
flags.DEFINE_integer('iterations_per_epoch', 100, 'Number of iteration per epoch')
flags.DEFINE_integer('train_batch_size', 64, 'Training batch size')
flags.DEFINE_integer('validation_batch_size', 100, 'Validation batch size')

flags.DEFINE_integer('save_freq', 10000, 'Save frequency')

NUM_CONV1_FILTERS = 64
NUM_CONV2_FILTERS = 64
NUM_CONV3_FILTERS = 128
NUM_CONV4_FILTERS = 128
NUM_CONV5_FILTERS = 256
NUM_CONV6_FILTERS = 256
NUM_CONV7_FILTERS = 256
NUM_CONV8_FILTERS = 256
NUM_CONV9_FILTERS = 512
NUM_CONV10_FILTERS = 512
NUM_CONV11_FILTERS = 512
NUM_CONV12_FILTERS = 512
NUM_CONV13_FILTERS = 512
NUM_CONV14_FILTERS = 512
NUM_CONV15_FILTERS = 512
NUM_CONV16_FILTERS = 512
NUM_FC17_UNITS = 4096
NUM_FC18_UNITS = 4096
NUM_FC19_UNITS = 1000

def linearize(x):
    x_shape = x.get_shape().as_list()
    x_length = x_shape[1] * x_shape[2] * x_shape[3]
    return tf.reshape(x, [-1, x_length]), x_length

def weight_and_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=DEFINE.random_normal_stddev))
    b = tf.Variable(tf.truncated_normal([shape[-1]], stddev=DEFINE.random_normal_stddev))
    return W, b

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

def model(x_in, dropout_keep_prob):
    x_2d = tf.reshape(x_in, [-1, IMAGE_SIZE_X, IMAGE_SIZE_Y, NUM_IMAGE_CHANNELS])

    w_conv1, b_conv1 = weight_and_bias([3, 3, NUM_IMAGE_CHANNELS, NUM_CONV1_FILTERS])
    w_conv2, b_conv2 = weight_and_bias([3, 3, NUM_CONV1_FILTERS, NUM_CONV2_FILTERS])
    conv1 = tf.nn.relu(tf.add(conv2d(x_2d, w_conv1), b_conv1))
    conv2 = tf.nn.relu(tf.add(conv2d(conv1, w_conv2), b_conv2))
    pool1 = max_pool(conv2)

    w_conv3, b_conv3 = weight_and_bias([3, 3, NUM_CONV2_FILTERS, NUM_CONV3_FILTERS])
    w_conv4, b_conv4 = weight_and_bias([3, 3, NUM_CONV3_FILTERS, NUM_CONV4_FILTERS])
    conv3 = tf.nn.relu(tf.add(conv2d(pool1, w_conv3), b_conv3))
    conv4 = tf.nn.relu(tf.add(conv2d(conv3, w_conv4), b_conv4))
    pool2 = max_pool(conv4)

    w_conv5, b_conv5 = weight_and_bias([3, 3, NUM_CONV4_FILTERS, NUM_CONV5_FILTERS])
    w_conv6, b_conv6 = weight_and_bias([3, 3, NUM_CONV5_FILTERS, NUM_CONV6_FILTERS])
    w_conv7, b_conv7 = weight_and_bias([3, 3, NUM_CONV6_FILTERS, NUM_CONV7_FILTERS])
    w_conv8, b_conv8 = weight_and_bias([3, 3, NUM_CONV7_FILTERS, NUM_CONV8_FILTERS])
    conv5 = tf.nn.relu(tf.add(conv2d(pool2, w_conv5), b_conv5))
    conv6 = tf.nn.relu(tf.add(conv2d(conv5, w_conv6), b_conv6))
    conv7 = tf.nn.relu(tf.add(conv2d(conv6, w_conv7), b_conv7))
    conv8 = tf.nn.relu(tf.add(conv2d(conv7, w_conv8), b_conv8))
    pool3 = max_pool(conv8)

    w_conv9, b_conv9 = weight_and_bias([3, 3, NUM_CONV8_FILTERS, NUM_CONV9_FILTERS])
    w_conv10, b_conv10 = weight_and_bias([3, 3, NUM_CONV9_FILTERS, NUM_CONV10_FILTERS])
    w_conv11, b_conv11 = weight_and_bias([3, 3, NUM_CONV10_FILTERS, NUM_CONV11_FILTERS])
    w_conv12, b_conv12 = weight_and_bias([3, 3, NUM_CONV11_FILTERS, NUM_CONV12_FILTERS])
    conv9 = tf.nn.relu(tf.add(conv2d(pool3, w_conv9), b_conv9))
    conv10 = tf.nn.relu(tf.add(conv2d(conv9, w_conv10), b_conv10))
    conv11 = tf.nn.relu(tf.add(conv2d(conv10, w_conv11), b_conv11))
    conv12 = tf.nn.relu(tf.add(conv2d(conv11, w_conv12), b_conv12))
    pool4 = max_pool(conv12)

    w_conv13, b_conv13 = weight_and_bias([3, 3, NUM_CONV12_FILTERS, NUM_CONV13_FILTERS])
    w_conv14, b_conv14 = weight_and_bias([3, 3, NUM_CONV13_FILTERS, NUM_CONV14_FILTERS])
    w_conv15, b_conv15 = weight_and_bias([3, 3, NUM_CONV14_FILTERS, NUM_CONV15_FILTERS])
    w_conv16, b_conv16 = weight_and_bias([3, 3, NUM_CONV15_FILTERS, NUM_CONV16_FILTERS])
    conv13 = tf.nn.relu(tf.add(conv2d(pool4, w_conv13), b_conv13))
    conv14 = tf.nn.relu(tf.add(conv2d(conv13, w_conv14), b_conv14))
    conv15 = tf.nn.relu(tf.add(conv2d(conv14, w_conv15), b_conv15))
    conv16 = tf.nn.relu(tf.add(conv2d(conv15, w_conv16), b_conv16))
    pool5 = max_pool(conv16)

    linear, linear_length = linearize(pool5)

    w_fc17, b_fc17 = weight_and_bias([linear_length, NUM_FC17_UNITS])
    fc17 = tf.nn.relu(tf.add(tf.matmul(linear, w_fc17), b_fc17))
    fc17 = tf.nn.dropout(fc17, dropout_keep_prob)

    w_fc18, b_fc18 = weight_and_bias([NUM_FC17_UNITS, NUM_FC18_UNITS])
    fc18 = tf.nn.relu(tf.add(tf.matmul(fc17, w_fc18), b_fc18))
    fc18 = tf.nn.dropout(fc18, dropout_keep_prob)

    w_fc19, b_fc19 = weight_and_bias([NUM_FC18_UNITS, NUM_FC19_UNITS])
    fc19 = tf.nn.relu(tf.add(tf.matmul(fc18, w_fc19), b_fc19))
    fc19 = tf.nn.dropout(fc19, dropout_keep_prob)

    w_out, b_out = weight_and_bias([NUM_FC19_UNITS, NUM_CLASS])
    return tf.nn.softmax(tf.add(tf.matmul(fc19, w_out), b_out))

def verify(y_in, y_out):
    return tf.equal(tf.argmax(y_out, 1), tf.argmax(y_in, 1))

def cross_entropy(y_in, y_out):
    return -tf.reduce_sum(y_in * tf.log(tf.clip_by_value(y_out, 1e-20, 1.0)))

def training(loss):
    return tf.train.AdamOptimizer(DEFINE.learning_rate).minimize(loss)

def main(argv):
    print("Preparing training data.")
    mnist = input_data.read_data_sets("./data/", validation_size=0, one_hot=True)

    x_in = tf.placeholder(tf.float32, [None, IMAGE_SIZE_X * IMAGE_SIZE_Y * NUM_IMAGE_CHANNELS])
    y_in = tf.placeholder(tf.float32, [None, NUM_CLASS])
    dropout_keep_prob = tf.placeholder(tf.float32)

    y_out = model(x_in, dropout_keep_prob)
    correct_prediction = verify(y_in, y_out)
    loss = cross_entropy(y_in, y_out)
    train = training(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Training started.")
    start = time.time()

    for i in range(DEFINE.epochs):
        loss_sum = 0.0
        for _ in range(DEFINE.iterations_per_epoch):
            batch_x, batch_y = mnist.train.next_batch(DEFINE.train_batch_size)
            loss_sum += sess.run([loss, train], feed_dict={x_in: batch_x, y_in: batch_y, dropout_keep_prob: DEFINE.dropout_keep_prob})[0]

        cp_sum = 0
        for _ in range(mnist.test.num_examples // DEFINE.validation_batch_size):
            batch_x, batch_y = mnist.test.next_batch(DEFINE.validation_batch_size)
            cp_sum += sess.run(correct_prediction, feed_dict={x_in: batch_x, y_in: batch_y, dropout_keep_prob: 1.0}).sum()

        print("{:08d}\t{:.15f}\t{:.4f}\t{:06d}".
              format(i + 1, loss_sum / DEFINE.iterations_per_epoch, cp_sum / mnist.test.num_examples, int(time.time() - start)))

if __name__ == '__main__':
    tf.app.run()