import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

file_dir = '/content/Cats_vs_Dogs/data/train/'  

label_list = []
image_list = []
for file in os.listdir(file_dir):
    image_list.append(file_dir+file)
    name = file.split(sep='.')
    if name[0] == 'cat':
        label_list.append([1, 0])
    else:
        label_list.append([0, 1])


image = tf.cast(image_list, tf.string)
label = tf.cast(label_list, tf.int32)

# make an input queue
input_queue = tf.train.slice_input_producer([image, label])

label = input_queue[1]
image_contents = tf.read_file(input_queue[0])
image = tf.image.decode_jpeg(image_contents, channels=3)
image = tf.image.per_image_standardization(image)
image = tf.image.resize_images(image, (224, 224), tf.image.ResizeMethod.BICUBIC)
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


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
# block1
W_block1_conv1 = weight_varible([3, 3, 3, 64])  # 3*3表示卷积核的尺寸，3表示图片的通道数量，64表示输出的通道
b_block1_conv1 = bias_variable([64])  # 每个输出通道上加偏置

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
h_block1_conv1 = tf.nn.relu(conv2d(x, W_block1_conv1) + b_block1_conv1)

W_block1_conv2 = weight_varible([3, 3, 64, 64])
b_block1_conv2 = bias_variable([64])

h_block1_conv2 = tf.nn.relu(conv2d(h_block1_conv1, W_block1_conv2) + b_block1_conv2)

block1_pool = max_pool_2x2(h_block1_conv2)

# block2

W_block2_conv1 = weight_varible([3, 3, 64, 128])  # 3*3表示卷积核的尺寸，3表示图片的通道数量，64表示输出的通道
b_block2_conv1 = bias_variable([128])  # 每个输出通道上加偏置

h_block2_conv1 = tf.nn.relu(conv2d(block1_pool, W_block2_conv1) + b_block2_conv1)

W_block2_conv2 = weight_varible([3, 3, 128, 128])
b_block2_conv2 = bias_variable([128])

h_block2_conv2 = tf.nn.relu(conv2d(h_block2_conv1, W_block2_conv2) + b_block2_conv2)

block2_pool = max_pool_2x2(h_block2_conv2)

# block3

W_block3_conv1 = weight_varible([3, 3, 128, 256])  # 3*3表示卷积核的尺寸，3表示图片的通道数量，64表示输出的通道
b_block3_conv1 = bias_variable([256])  # 每个输出通道上加偏置

h_block3_conv1 = tf.nn.relu(conv2d(block2_pool, W_block3_conv1) + b_block3_conv1)

W_block3_conv2 = weight_varible([3, 3, 256, 256])
b_block3_conv2 = bias_variable([256])

h_block3_conv2 = tf.nn.relu(conv2d(h_block3_conv1, W_block3_conv2) + b_block3_conv2)

block3_pool = max_pool_2x2(h_block3_conv2)

# block4

W_block4_conv1 = weight_varible([3, 3, 256, 512])  # 3*3表示卷积核的尺寸，3表示图片的通道数量，64表示输出的通道
b_block4_conv1 = bias_variable([512])  # 每个输出通道上加偏置

h_block4_conv1 = tf.nn.relu(conv2d(block3_pool, W_block4_conv1) + b_block4_conv1)

W_block4_conv2 = weight_varible([3, 3, 512, 512])
b_block4_conv2 = bias_variable([512])

h_block4_conv2 = tf.nn.relu(conv2d(h_block4_conv1, W_block4_conv2) + b_block4_conv2)

block4_pool = max_pool_2x2(h_block4_conv2)

# block5

W_block5_conv1 = weight_varible([3, 3, 512, 512])  # 3*3表示卷积核的尺寸，3表示图片的通道数量，64表示输出的通道
b_block5_conv1 = bias_variable([512])  # 每个输出通道上加偏置

h_block5_conv1 = tf.nn.relu(conv2d(block4_pool, W_block5_conv1) + b_block5_conv1)

W_block5_conv2 = weight_varible([3, 3, 512, 512])
b_block5_conv2 = bias_variable([512])

h_block5_conv2 = tf.nn.relu(conv2d(h_block5_conv1, W_block5_conv2) + b_block5_conv2)

block5_pool = max_pool_2x2(h_block5_conv2)


# full connection
W_fc1 = weight_varible([7 * 7 * 512, 4096])
b_fc1 = bias_variable([4096])
h_pool2_flat = tf.reshape(block5_pool, [-1, 7 * 7 * 512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# output layer: softmax
W_fc2 = weight_varible([4096, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 2])

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
    while not coord.should_stop() and i < 500:

        img, label = sess.run([image_batch, label_batch])
        train_accuacy = accuracy.eval(feed_dict={x: img, y_: label, keep_prob: 1})
        print("step %d, training accuracy %g" % (i, train_accuacy))
        train_step.run(feed_dict={x: img, y_: label, keep_prob: 0.5})
        i += 1


except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
coord.join(threads)
saver.save(sess, "./model/model-conv", global_step=i)







