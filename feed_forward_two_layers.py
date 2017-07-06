import tensorflow as tf

'''
Two layers feed forward model
'''


class Feed_forward_two_layers:
    def get_name(self):
        return "feed_forward_two_layers"

    def input_placeholders(self):
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        return inputs_placeholder, labels_placeholder

    def inference(self, input):
        with tf.name_scope("inference"):
            middle = self._fully_connected(input, 784, 64)
            logits = self._fully_connected(middle, 64, 10)
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
