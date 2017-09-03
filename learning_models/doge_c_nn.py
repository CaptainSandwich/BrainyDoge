import tensorflow as tf
import glob
import imghdr
import os
import numpy as np
from tqdm import *

record_queue = tf.train.string_input_producer(glob.glob('../Data/labels'), num_epochs=1)
record_reader = tf.TextLineReader()
record_key, record = record_reader.read(record_queue)
image_name, label = tf.decode_csv(record, record_defaults=[[""], [0]])

file_name = tf.placeholder(tf.string)
read_image = tf.read_file(image_name)
decoded_image = tf.image.decode_png(read_image, channels=3, dtype=tf.uint8)
dimensions = tf.placeholder(tf.int32)
resized_image = tf.image.resize_images(decoded_image, dimensions)

training_set = tf.placeholder(tf.float32, [None, 10800])
image_queue = tf.FIFOQueue(capacity=50, dtypes=tf.float32)
image_enqueue = image_queue.enqueue(resized_image)
image_dequeue = image_queue.dequeue()

y_ = tf.placeholder(tf.int32)
label_queue = tf.FIFOQueue(capacity=50, dtypes=tf.int32)
label_enqueue = label_queue.enqueue(label)
label_dequeue = label_queue.dequeue()

W = tf.Variable(tf.zeros([10800, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(training_set, W) + b

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.cast(y, tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        with tf.device('/gpu:0'):
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            image_list = []
            label_list = []
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
                try:
                    sess.run([image_enqueue, label_enqueue], feed_dict={dimensions: [60, 60]})
                    img_data, img_label = sess.run([image_dequeue, label_dequeue])
                    image_list.append(img_data)
                    label_list.append(img_label)
                except tf.errors.InvalidArgumentError as e:
                    print("Arg error")
                except ValueError as e:
                    print("Value error")
                except tf.errors.OutOfRangeError:
                    print('Finished loading images')
                    coord.request_stop()
            coord.join(threads)

            for i in tqdm(range(1000)):
                sess.run(train_step, feed_dict={training_set: np.asarray(image_list).reshape(-1, 10800), y_: label_list})

            print(sess.run(accuracy, feed_dict={training_set: np.asarray(image_list).reshape(-1, 10800), y_: label_list}))

            sess.close()
