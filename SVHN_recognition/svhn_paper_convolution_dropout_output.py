import tensorflow as tf

'''
Model for sequence classification and localization with weighted loss
'''


class SVHNPaperConvolutionDropoutOutput:
    def get_name(self):
        return "SVHN_paper_convolution_dropout_output_0.8"

    def input_placeholders(self):
        inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 128, 256, 3], name="inputs")
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 6, 11], name="labels")
        positions_placeholder = tf.placeholder(tf.float32, shape=[None, 4], name="positions")
        keep_prob_placeholder = tf.placeholder(tf.float32)
        is_training_placeholder = tf.placeholder(tf.bool)
        return inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder, is_training_placeholder

    def inference(self, input, keep_prob, is_training):
        with tf.name_scope("inference"):
            input = tf.reshape(input, [-1, 128, 256, 3])
            conv1 = self._convolutional(input, [5, 5, 3, 48])
            relu1 = self._relu(conv1)
            max_pool1 = self._max_pooling(relu1, [1, 2, 2, 1], [1, 2, 2, 1])

            conv2 = self._convolutional(max_pool1, [5, 5, 48, 64])
            relu2 = self._relu(conv2)
            max_pool2 = self._max_pooling(relu2, [1, 2, 2, 1], [1, 2, 2, 1])

            conv3 = self._convolutional(max_pool2, [5, 5, 64, 128])
            relu3 = self._relu(conv3)
            max_pool3 = self._max_pooling(relu3, [1, 2, 2, 1], [1, 2, 2, 1])

            conv4 = self._convolutional(max_pool3, [5, 5, 128, 160])
            relu4 = self._relu(conv4)
            max_pool4 = self._max_pooling(relu4, [1, 2, 2, 1], [1, 2, 2, 1])

            conv5 = self._convolutional(max_pool4, [2, 2, 160, 192])
            relu5 = self._relu(conv5)
            max_pool5 = self._max_pooling(relu5, [1, 2, 2, 1], [1, 2, 2, 1])

            conv6 = self._convolutional(max_pool5, [2, 2, 192, 192])
            relu6 = self._relu(conv6)
            max_pool6 = self._max_pooling(relu6, [1, 2, 2, 1], [1, 2, 2, 1])

            reshaped = tf.reshape(max_pool6, [-1, 1536])

            logits = []

            with tf.variable_scope("Linear_output"):
                dropout_one = tf.nn.dropout(reshaped, keep_prob)
                label_one = self._fully_connected(dropout_one, 1536, 11, name="label_one")

                dropout_two = tf.nn.dropout(reshaped, keep_prob)
                label_two = self._fully_connected(dropout_two, 1536, 11, name="label_two")

                dropout_three = tf.nn.dropout(reshaped, keep_prob)
                label_three = self._fully_connected(dropout_three, 1536, 11, name="label_three")

                dropout_four = tf.nn.dropout(reshaped, keep_prob)
                label_four = self._fully_connected(dropout_four, 1536, 11, name="label_four")

                dropout_five = tf.nn.dropout(reshaped, keep_prob)
                label_five = self._fully_connected(dropout_five, 1536, 11, name="label_five")

                dropout_six = tf.nn.dropout(reshaped, keep_prob)
                label_six = self._fully_connected(dropout_six, 1536, 11, name="label_six")

                logits.append(label_one)
                logits.append(label_two)
                logits.append(label_three)
                logits.append(label_four)
                logits.append(label_five)
                logits.append(label_six)

            dropout_position = tf.nn.dropout(reshaped, keep_prob)
            predicted_positions = self._fully_connected(dropout_position, 1536, 4)
            return tf.identity(tf.stack(logits, axis=1), name="labels"), tf.identity(predicted_positions,
                                                                                     name="position")

    def loss(self, logits, labels, predicted_positions, positions):
        with tf.name_scope("loss"):
            labels = tf.to_int64(labels)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="cross_entropy")
            logits_loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

            square_error = tf.square(positions - predicted_positions, name="square_error")
            position_loss = tf.reduce_mean(square_error, name="square_error_mean")

            total_loss = 1000 * logits_loss + position_loss

            tf.summary.scalar("logits_loss", logits_loss)
            tf.summary.scalar("positions_loss", position_loss)
            tf.summary.scalar("total_loss", logits_loss + position_loss)
            return {"logits_loss": logits_loss, "positions_loss": position_loss,
                    "total_loss": total_loss}

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

            return self.tf_count(corrects,
                                 True), corrects, logits, position_error, predicted_positions, total_correct_characters, total_characters

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
