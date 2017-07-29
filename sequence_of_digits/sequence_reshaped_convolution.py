import tensorflow as tf

'''
Recurrent model for sequence recognition with dropout
'''


class SequenceReshapedConvolution:
    def get_name(self):
        return "sequence_reshaped_convolution"

    def input_placeholders(self):
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 140], name="inputs")
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 5, 10], name="labels")
        keep_prob_placeholder = tf.placeholder(tf.float32)
        return inputs_placeholder, labels_placeholder, keep_prob_placeholder

    def inference(self, input, keep_prob, is_training):
        with tf.name_scope("inference"):
            input = tf.reshape(input, [-1, 28, 140, 1])
            conv1 = self._convolutional(input, [10, 10, 1, 8])
            relu1 = self._relu(conv1)
            dropout1 = tf.nn.dropout(relu1, keep_prob)
            max_pool1 = self._max_pooling(dropout1, [1, 2, 2, 1], [1, 2, 2, 1])

            conv2 = self._convolutional(max_pool1, [5, 5, 8, 16])
            relu2 = self._relu(conv2)
            dropout2 = tf.nn.dropout(relu2, keep_prob)
            max_pool2 = self._max_pooling(dropout2, [1, 2, 2, 1], [1, 2, 2, 1])

            conv3 = self._convolutional(max_pool2, [2, 2, 16, 32])
            relu3 = self._relu(conv3)
            dropout3 = tf.nn.dropout(relu3, keep_prob)
            max_pool3 = self._max_pooling(dropout3, [1, 2, 2, 1], [1, 2, 2, 1])

            reshaped = tf.reshape(max_pool3, [-1, 2304])

            logits = []
            gru = tf.contrib.rnn.GRUCell(576)
            state = gru.zero_state(tf.shape(reshaped)[0], tf.float32)
            with tf.variable_scope("RNN"):
                for i in range(5):
                    if i > 0: tf.get_variable_scope().reuse_variables()
                    output, state = gru(reshaped, state)
                    number_logits = self._fully_connected(output, 576, 10)
                    logits.append(number_logits)
            return tf.stack(logits, axis=1)

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
            labels = tf.argmax(labels, 2)
            logits = tf.argmax(logits, 2)
            difference = tf.subtract(labels, logits, name="sub")
            corrects = tf.count_nonzero(difference, axis=1, name="count_nonzero")
            corrects = tf.less_equal(corrects, 0, name="is_zero")

            return self.tf_count(corrects, True), corrects, logits

    def tf_count(self, t, val):
        elements_equal_to_value = tf.equal(t, val)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        return count

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
