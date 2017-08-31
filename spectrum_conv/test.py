import tensorflow as tf
import numpy as np
import os
import pandas as pd

file_dir = './train/JPEG/'

angry = []
label_angry = []
happy = []
label_happy = []
neutral = []
label_neutral = []
sad = []
label_sad = []
for file in os.listdir(file_dir):
    name = file.split(sep='-')
    if name[0] == 'angry':
        angry.append(file_dir + file)
        label_angry.append([1, 0, 0, 0])
    elif name[0] == 'happy':
        happy.append(file_dir + file)
        label_happy.append([0, 1, 0, 0])
    elif name[0] == 'neutral':
        neutral.append(file_dir + file)
        label_neutral.append([0, 0, 1, 0])
    elif name[0] == 'sad':
        sad.append(file_dir + file)
        label_sad.append([0, 0, 0, 1])
print('There are %d angry samples\nThere are %d happy samples\nThere are %d neutral samples\nThere are %d sad samples\n' % (len(angry), len(happy), len(neutral), len(sad)))
image_list = []
label_list = []
image_list.extend(angry)
image_list.extend(happy)
image_list.extend(neutral)
image_list.extend(sad)
label_list.extend(label_angry)
label_list.extend(label_happy)
label_list.extend(label_neutral)
label_list.extend(label_sad)
print(len(label_list), len(image_list))

image = tf.cast(image_list, tf.string)
label = tf.cast(label_list, tf.int32)

# make an input queue
input_queue = tf.train.slice_input_producer([image, label])

label = input_queue[1]
image_contents = tf.read_file(input_queue[0])
image = tf.image.decode_jpeg(image_contents, channels=3)
image = tf.image.per_image_standardization(image)
image = tf.image.resize_image_with_crop_or_pad(image, 23, 44)
######################################
# data argumentation should go to here
######################################
image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size=4798,
                                                num_threads=64,
                                                capacity=20000,
                                                min_after_dequeue=20000 - 1)

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 0和3是默认，1和2表示x轴和y轴的步长

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()
# conv_layer1
W_conv1 = weight_varible([2, 2, 3, 32])  # 2x2表示卷积核的尺寸，1表示图片的通道数量，32表示输出的通道
b_conv1 = bias_variable([32])  # 每个输出通道上加偏置

x = tf.placeholder(tf.float32, [None, 23, 44, 3])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv_layer2
W_conv2 = weight_varible([2, 2, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# full connection
W_fc1 = weight_varible([11 * 6 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 11 * 6 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# output layer: softmax
W_fc2 = weight_varible([1024, 4])
b_fc2 = bias_variable([4])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 4])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
msort = tf.arg_max(y_conv, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
i = 0
saver.restore(sess, "./model/model-conv-5000")
try:
    while not coord.should_stop() and i < 1:

        img, label = sess.run([image_batch, label_batch])
        train_accuacy = accuracy.eval(feed_dict={x: img, y_: label})
        print("Accuracy %g" % (train_accuacy))
        temp = tf.arg_max(label, 1)
        qiwang = temp.eval()
        qiwang = list(qiwang)
        jieguo = msort.eval(feed_dict={x: img})
        jieguo = list(jieguo)
        Confusion_Matrix = np.zeros((4, 4), dtype=np.int32)
        true_angry = 0
        true_happy = 0
        true_neutral = 0
        true_sad = 0
        for mi in range(len(qiwang)):
            if qiwang[mi] == 0:
                true_angry += 1
                if jieguo[mi] == 0:
                    Confusion_Matrix[0][0] += 1
                elif jieguo[mi] == 1:
                    Confusion_Matrix[0][1] += 1
                elif jieguo[mi] == 2:
                    Confusion_Matrix[0][2] += 1
                elif jieguo[mi] == 3:
                    Confusion_Matrix[0][3] += 1
            if qiwang[mi] == 1:
                true_happy += 1
                if jieguo[mi] == 1:
                    Confusion_Matrix[1][1] += 1
                elif jieguo[mi] == 0:
                    Confusion_Matrix[1][0] += 1
                elif jieguo[mi] == 2:
                    Confusion_Matrix[1][2] += 1
                elif jieguo[mi] == 3:
                    Confusion_Matrix[1][3] += 1
            if qiwang[mi] == 2:
                true_neutral += 1
                if jieguo[mi] == 2:
                    Confusion_Matrix[2][2] += 1
                elif jieguo[mi] == 0:
                    Confusion_Matrix[2][0] += 1
                elif jieguo[mi] == 1:
                    Confusion_Matrix[2][1] += 1
                elif jieguo[mi] == 3:
                    Confusion_Matrix[2][3] += 1
            if qiwang[mi] == 3:
                true_sad += 1
                if jieguo[mi] == 3:
                    Confusion_Matrix[3][3] += 1
                elif jieguo[mi] == 0:
                    Confusion_Matrix[3][0] += 1
                elif jieguo[mi] == 1:
                    Confusion_Matrix[3][1] += 1
                elif jieguo[mi] == 2:
                    Confusion_Matrix[3][2] += 1
        i += 1


except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
coord.join(threads)
print("True angry:", true_angry)
print("True happy", true_happy)
print("True neutral:", true_neutral)
print("True sad:", true_sad)
print("\nConfusion_Matrix\n")
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









