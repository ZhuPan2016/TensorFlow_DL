import tensorflow as tf
import numpy as np
import os

file_dir = './train/JPEG/'  # 链接: https://pan.baidu.com/s/1i4De02x 密码: hcm4

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
print('There are %d angry\nThere are %d happy\nThere are %d neutral\nThere are %d sad\n' % (len(angry), len(happy), len(neutral), len(sad)))
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
                                                batch_size=100,
                                                num_threads=64,
                                                capacity=2000,
                                                min_after_dequeue=2000 - 1)

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#0和3是默认，1和2表示x轴和y轴的步长

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()
# conv_layer1
W_conv1 = weight_varible([2, 2, 3, 32])#2x2表示卷积核的尺寸，1表示图片的通道数量，32表示输出的通道
b_conv1 = bias_variable([32])#每个输出通道上加偏置

x = tf.placeholder(tf.float32, [None, 23, 44, 3])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#conv_layer2
W_conv2 = weight_varible([2, 2, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# full connection
W_fc1 = weight_varible([11 * 6 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 11 * 6 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# output layer: softmax
W_fc2 = weight_varible([1024, 4])
b_fc2 = bias_variable([4])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 4])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
i = 0
try:
    while not coord.should_stop() and i < 5000:

        img, label = sess.run([image_batch, label_batch])
        train_accuacy = accuracy.eval(feed_dict={x: img, y_: label, keep_prob: 0.5})
        print("step %d, training accuracy %g" % (i, train_accuacy))
        train_step.run(feed_dict={x: img, y_: label, keep_prob: 0.5})
        i += 1


except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
coord.join(threads)
saver.save(sess, "./model/model-conv", global_step=i)







