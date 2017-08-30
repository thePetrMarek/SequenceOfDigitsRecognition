import tensorflow as tf
import numpy as np
from visualize import Visualize
import scipy.misc

image = "image.jpg"


def to_label(label):
    text_label = ""
    for single_label in label:
        number = np.argmax(single_label)
        if number == 10:
            return text_label
        else:
            text_label += str(number)
    return text_label


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('SVHN_recognition/checkpoints/SVHN/SVHN-30000.meta')
    saver.restore(sess, 'SVHN_recognition/checkpoints/SVHN/SVHN-30000')
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name("inputs:0")
    label = graph.get_tensor_by_name("inference/stack:0")
    position = graph.get_tensor_by_name("inference/fc_5/MatMul:0")

    input = scipy.misc.imresize(scipy.misc.imread(image), (128, 256))

    feed_dict = {inputs: [input]}
    label, position = sess.run([label, position], feed_dict)
    visualize = Visualize()
    visualize.visualize_inference(input, to_label(label[0]), position[0])
