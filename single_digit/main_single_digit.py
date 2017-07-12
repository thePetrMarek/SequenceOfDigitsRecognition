import tensorflow as tf
from convolutional import Convolutional
from tensorflow.examples.tutorials.mnist import input_data

'''
Main file for running of single digit recognition models
'''

def get_batch(dataset, inputs_placeholder, labels_placeholder):
    inputs, labels = dataset.next_batch(50)
    return {inputs_placeholder: inputs, labels_placeholder: labels}


def evaluate(dataset, session, operation, inputs_placeholder, labels_placeholder, name, summary_writer, learning_step):
    steps_per_epoch = dataset.num_examples // 50
    number_of_examples = steps_per_epoch * 50

    correct_num = 0
    for step in range(steps_per_epoch):
        batch = get_batch(dataset, inputs_placeholder, labels_placeholder)
        correct_num += session.run(operation, feed_dict=batch)

    precision = correct_num / number_of_examples
    summary = tf.Summary()
    summary.value.add(tag='Accuracy_' + name, simple_value=precision)
    summary_writer.add_summary(summary, learning_step)
    print("Accuracy %.3f" % precision)


if __name__ == '__main__':

    # Download mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Graph().as_default():
        # Wiring and different models
        #model = Feed_forward()
        #model = Feed_forward_two_layers()
        model = Convolutional()
        inputs_placeholder, labels_placeholder = model.input_placeholders()
        logits = model.inference(inputs_placeholder)
        loss = model.loss(logits, labels_placeholder)
        training = model.training(loss, 0.0001)
        evaluation = model.evaluation(logits, labels_placeholder)

        # Initialization
        session = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        session.run(init)

        # visualize graph
        writer = tf.summary.FileWriter("visualizations/" + model.get_name())
        writer.add_graph(session.graph)

        # Summaries
        merged_summary = tf.summary.merge_all()

        # Training
        for step in range(10000 + 1):
            batch = get_batch(mnist.train, inputs_placeholder, labels_placeholder)
            loss_value, summary, _ = session.run([loss, merged_summary, training], feed_dict=batch)
            writer.add_summary(summary, step)
            if step % 100 == 0:
                print("Step %d, loss %.3f" % (step, loss_value))
                print("Train accuracy")
                evaluate(mnist.train, session, evaluation, inputs_placeholder, labels_placeholder, "train", writer,
                         step)
                print("Validation accuracy")
                evaluate(mnist.validation, session, evaluation, inputs_placeholder, labels_placeholder, "validation",
                         writer, step)
                print("Test accuracy")
                evaluate(mnist.test, session, evaluation, inputs_placeholder, labels_placeholder, "test", writer, step)
                print()
