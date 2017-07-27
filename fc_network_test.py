# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
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


def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 22])
#layer 1
W_fc1 = weight_varible([22, 200])
b_fc1 = bias_variable([200])

h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_fc1) + b_fc1)
# layer 2
W_fc2 = weight_varible([200, 200])
b_fc2 = bias_variable([200])

h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

#output
outWeight = weight_varible([200, 4])
outB = bias_variable([4])
y_out = tf.nn.softmax(tf.matmul(h_fc2, outWeight) + outB)


# model training
y_ = tf.placeholder(tf.float32, [None, 4])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_out), reduction_indices=[1])
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_out, 1), tf.arg_max(y_, 1))
msort = tf.arg_max(y_out, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
saver.restore(sess, "./model/model.ckpt-2627")
for i in range(1):
    features_list = []
    label_list = []
    for xx in range(1642):
        example, biaoqian= sess.run([features, labels])
        features_list.append(example)
        label_list.append(biaoqian)

    features_list = np.array(features_list)
    features_list = features_list# + np.random.normal(loc=0.0, scale=0.5, size=features_list.shape)
    label_list = np.array(label_list)

    shuchu = y_out.eval(feed_dict={x: features_list, y_: label_list})

    train_accuacy = accuracy.eval(feed_dict={x: features_list, y_: label_list})
    print("step %d, training accuracy %g" % (i, train_accuacy))
    temp = tf.arg_max(label_list, 1)
    qiwang = temp.eval()
    qiwang = list(qiwang)
    jieguo = msort.eval(feed_dict={x: features_list, y_: label_list})
    jieguo= list(jieguo)
    Confusion_Matrix = np.zeros((4, 4), dtype=np.int32)
    for mi in range(len(qiwang)):
        if qiwang[mi] == 0:
            if jieguo[mi] == 0:
                Confusion_Matrix[0][0] += 1
            elif jieguo[mi] == 1:
                Confusion_Matrix[0][1] += 1
            elif jieguo[mi] == 2:
                Confusion_Matrix[0][2] += 1
            elif jieguo[mi] == 3:
                Confusion_Matrix[0][3] += 1
        if qiwang[mi] == 1:
            if jieguo[mi] ==1:
                Confusion_Matrix[1][1] += 1
            elif jieguo[mi] == 0:
                Confusion_Matrix[1][0] += 1
            elif jieguo[mi] == 2:
                Confusion_Matrix[1][2] += 1
            else:
                Confusion_Matrix[1][3] += 1
        if qiwang[mi] == 2:
            if jieguo[mi] == 2:
                Confusion_Matrix[2][2] += 1
            elif jieguo[mi] == 0:
                Confusion_Matrix[2][0] += 1
            elif jieguo[mi] == 1:
                Confusion_Matrix[2][1] += 1
            elif jieguo[mi] == 3:
                Confusion_Matrix[2][3] += 1
        if qiwang[mi] == 3:
            if jieguo[mi] == 3:
                Confusion_Matrix[3][3] += 1
            elif jieguo[mi] == 0:
                Confusion_Matrix[3][0] += 1
            elif jieguo[mi] == 1:
                Confusion_Matrix[3][1] += 1
            else:
                Confusion_Matrix[3][2] += 1
coord.request_stop()
coord.join(threads)
print(Confusion_Matrix)
Confusion_Matrix_sum =Confusion_Matrix.sum(axis=1)
acc = []
for m in range(4):
    acc.append(Confusion_Matrix[m][m]/Confusion_Matrix_sum[m])
DATA = pd.DataFrame(Confusion_Matrix, index=['Actual angry', 'Actual happy', 'Actual neutral', 'Actual sad', ], columns=
                    ['Predict angry', 'Pre happy', 'Pre neutral', 'Pre sad'])

DATA['sum'] = Confusion_Matrix_sum
DATA['ACC'] = acc
print(DATA)

