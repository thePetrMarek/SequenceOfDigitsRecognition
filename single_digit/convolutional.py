import tensorflow as tf

'''
Convolutional model
'''


class Convolutional:
    def get_name(self):
        return "convolutional"

    def input_placeholders(self):
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        return inputs_placeholder, labels_placeholder

    def inference(self, input):
        with tf.name_scope("inference"):
            input = tf.reshape(input, [-1, 28, 28, 1])
            conv1 = self._convolutional(input, [5, 5, 1, 32])
            relu1 = self._relu(conv1)
            max_pool1 = self._max_pooling(relu1, [1, 2, 2, 1], [1, 2, 2, 1])

            conv2 = self._convolutional(max_pool1, [3, 3, 32, 16])
            relu2 = self._relu(conv2)
            max_pool2 = self._max_pooling(relu2, [1, 2, 2, 1], [1, 2, 2, 1])

            conv3 = self._convolutional(max_pool2, [2, 2, 16, 8])
            relu3 = self._relu(conv3)
            max_pool3 = self._max_pooling(relu3, [1, 2, 2, 1], [1, 2, 2, 1])

            reshaped = tf.reshape(max_pool3, [-1, 128])
            fc1 = self._fully_connected(reshaped, 128, 64)
            fc2 = self._fully_connected(fc1, 64, 32)
            logits = self._fully_connected(fc2, 32, 10)

            return logits

    def loss(self, logits, labels):
        with tf.name_scope("loss"):
            labels = tf.to_int64(labels)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                    name="cross_entropy")
            mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
            tf.summary.scalar("loss", mean)
            return mean

    def training(self, loss, learning_rate):
        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(loss)
            return train_operation

    def evaluation(self, logits, labels):
        with tf.name_scope("evaluation"):
            labels = tf.to_int64(labels)
            labels = tf.argmax(labels, 1)
            correct = tf.nn.in_top_k(logits, labels, 1)
            return tf.reduce_sum(tf.cast(correct, tf.int32))

    def _fully_connected(self, input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
            act = tf.matmul(input, w) + b
            return act

    def _convolutional(self, input, dimensions, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(dimensions, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[dimensions[3]]), name="b")
            return tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b

    def _max_pooling(self, input, ksize, strides, name="max_pooling"):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize, strides, padding="SAME")

    def _relu(self, input, name="relu"):
        with tf.name_scope(name):
            return tf.nn.relu(input)
