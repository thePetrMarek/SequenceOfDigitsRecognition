import tensorflow as tf

'''
Model for sequence classification and localization
'''


class SequenceLocalization:
    def get_name(self):
        return "sequence_localization"

    def input_placeholders(self):
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 256], name="inputs")
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 5, 10], name="labels")
        positions_placeholder = tf.placeholder(tf.float32, shape=[None, 4], name="positions")
        keep_prob_placeholder = tf.placeholder(tf.float32)
        is_training_placeholder = tf.placeholder(tf.bool)
        return inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder, is_training_placeholder

    def inference(self, input, keep_prob, is_training):
        with tf.name_scope("inference"):
            input = tf.reshape(input, [-1, 128, 256, 1])
            conv1 = self._convolutional(input, [3, 3, 1, 6])
            relu1 = self._relu(conv1)
            dropout1 = tf.nn.dropout(relu1, keep_prob)
            max_pool1 = self._max_pooling(dropout1, [1, 2, 2, 1], [1, 2, 2, 1])

            conv2 = self._convolutional(max_pool1, [3, 3, 6, 10])
            relu2 = self._relu(conv2)
            dropout2 = tf.nn.dropout(relu2, keep_prob)
            max_pool2 = self._max_pooling(dropout2, [1, 2, 2, 1], [1, 2, 2, 1])

            conv3 = self._convolutional(max_pool2, [3, 3, 10, 16])
            relu3 = self._relu(conv3)
            dropout3 = tf.nn.dropout(relu3, keep_prob)
            max_pool3 = self._max_pooling(dropout3, [1, 2, 2, 1], [1, 2, 2, 1])

            reshaped = tf.reshape(max_pool3, [-1, 8192])

            fully1 = self._fully_connected(reshaped, 8192, 4096)
            relu4 = self._relu(fully1)
            fully2 = self._fully_connected(relu4, 4096, 576)

            logits = []
            gru = tf.contrib.rnn.GRUCell(576)
            state = gru.zero_state(tf.shape(fully2)[0], tf.float32)
            with tf.variable_scope("RNN"):
                for i in range(5):
                    if i > 0: tf.get_variable_scope().reuse_variables()
                    output, state = gru(fully2, state)
                    number_logits = self._fully_connected(output, 576, 10)
                    logits.append(number_logits)

            fc_position = self._fully_connected(fully2, 576, 128)
            relu_position = self._relu(fc_position)
            predicted_positions = self._fully_connected(relu_position, 128, 4)
            return tf.stack(logits, axis=1), predicted_positions

    def loss(self, logits, labels, predicted_positions, positions):
        with tf.name_scope("loss"):
            labels = tf.to_int64(labels)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                    name="cross_entropy")
            logits_loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

            position_loss = tf.losses.mean_squared_error(positions, predicted_positions)

            tf.summary.scalar("logits_loss", logits_loss)
            tf.summary.scalar("positions_loss", position_loss)
            tf.summary.scalar("total_loss", logits_loss + position_loss)
            return {"logits_loss": logits_loss, "positions_loss": position_loss,
                    "total_loss": logits_loss + position_loss}

    def training(self, loss, learning_rate):
        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(loss)
            return train_operation

    def evaluation(self, logits, labels, predicted_positions, positions):
        with tf.name_scope("evaluation"):
            labels = tf.to_int64(labels)
            labels = tf.argmax(labels, 2)
            logits = tf.argmax(logits, 2)
            difference = tf.subtract(labels, logits, name="sub")
            character_errors = tf.count_nonzero(difference, axis=1, name="count_nonzero")
            total_wrong_characters = tf.reduce_sum(character_errors)
            total_characters = tf.to_int64(tf.size(labels))
            total_correct_characters = total_characters - total_wrong_characters
            corrects = tf.less_equal(character_errors, 0, name="is_zero")

            position_error = tf.losses.mean_squared_error(positions, predicted_positions)

            return self.tf_count(corrects, True), corrects, logits, position_error, predicted_positions, total_correct_characters, total_characters

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
