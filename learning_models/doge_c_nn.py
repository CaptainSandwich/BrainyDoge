import tensorflow as tf
import glob
import time
import os


def load_images(sess):
    start = time.time()

    img_list = []
    i = 0

    filename_queue = tf.train.string_input_producer(glob.glob('../Data/doges/*.png'), num_epochs=1)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_image(image_file, 3)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while not coord.should_stop():
        try:
            img = sess.run(image)
            img_list.append(img)
            i += 1
        except tf.errors.InvalidArgumentError:
            print('Invalid image data')
            i += 1
        except tf.errors.OutOfRangeError:
            print('done')
            coord.request_stop()

    coord.request_stop()
    coord.join(threads)

    end = time.time()
    print('Loading time: ' + str(end - start) + 's')


with tf.Session() as session:
    load_images(session)

    session.close()
