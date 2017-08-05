# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
filedir = '4emotionsLabledVec_shuffle.csv'
filename_queue = tf.train.string_input_producer([filedir])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                       [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, \
     col19, col20, col21, col22, col23, col24, col25, col26 = tf.decode_csv(
        value, record_defaults=record_defaults, field_delim=',')
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15,
                         col16, col17, col18, col19, col20, col21, col22])
labels = tf.stack([col23, col24, col25, col26])
# 以上几句代码写的异常暴力，可以用生成器表达式写的非常优雅，但是现在是在机器学习里头，要什么自行车...


def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()
# fully connected layer 1
x = tf.placeholder(tf.float32, [None, 22])
W_fc1 = weight_varible([22, 200])  # 输入是22个特征，结点有200个
b_fc1 = bias_variable([200])  # 偏置
h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# fully connected layer 2
W_fc2 = weight_varible([200, 200])
b_fc2 = bias_variable([200])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
# dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
# output
outWeight = weight_varible([200, 4])
outB = bias_variable([4])
y_out = tf.nn.softmax(tf.matmul(h_fc2_drop, outWeight) + outB)
# model training
y_ = tf.placeholder(tf.float32, [None, 4])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_out), reduction_indices=[1])
train_step = tf.train.AdamOptimizer(1e-3,).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_out, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for i in range(10000):
    features_list = []
    label_list = []
    for xx in range(164 * 3):
        example, label = sess.run([features, labels])
        features_list.append(example)
        label_list.append(label)
    features_list = np.array(features_list)
    label_list = np.array(label_list)

    train_accuacy = accuracy.eval(feed_dict={x: features_list, y_: label_list, keep_prob: 0.75})
    print("step %d, training accuracy %g" % (i, train_accuacy))
    train_step.run(feed_dict={x: features_list, y_: label_list, keep_prob: 0.75})
    if i > 9998:
        saver.save(sess, "./model/model", global_step=i)
coord.request_stop()
coord.join(threads)

