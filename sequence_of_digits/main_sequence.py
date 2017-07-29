import os

import numpy as np
import tensorflow as tf

from prepare_dataset import load_dataset
from sequence_of_digits.sequence_reshaped_convolution_batchnorm import SequenceReshapedConvolutionBatchnorm
from visualize import Visualize


def get_batch(dataset, inputs_placeholder, labels_placeholder, keep_prob_placeholder, keep_prob_val,
              is_training_placeholder, is_traininig):
    if "position" not in dataset:
        dataset["position"] = 0
    position = dataset["position"]
    steps_per_epoch = len(dataset['examples']) // 50
    inputs = dataset['examples'][50 * position: (50 * position) + 50]
    labels = dataset['labels'][50 * position: (50 * position) + 50]
    position += 1
    if position == steps_per_epoch:
        position = 0
    dataset["position"] = position
    return {inputs_placeholder: inputs, labels_placeholder: labels, keep_prob_placeholder: keep_prob_val,
            is_training_placeholder: is_traininig}


def evaluate(dataset, session, operation, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
             is_training_placeholder, name,
             summary_writer, learning_step, visualize_correct=0, visualize_incorrect=0):
    steps_per_epoch = len(dataset['examples']) // 50
    number_of_examples = steps_per_epoch * 50

    visualize = Visualize()
    correct_visualized_counter = 0
    incorrect_visualized_counter = 0

    correct_num = 0
    for step in range(steps_per_epoch):
        batch = get_batch(dataset, inputs_placeholder, labels_placeholder, keep_prob_placeholder, 1,
                          is_training_placeholder, False)
        corrects_in_batch, corrects_vector, predictions = session.run(operation, feed_dict=batch)
        correct_num += corrects_in_batch

        # visualize correct and incorrect recognitions
        if incorrect_visualized_counter < visualize_incorrect or correct_visualized_counter < visualize_correct:
            for i in range(len(batch[inputs_placeholder])):
                true_label = np.argmax(batch[labels_placeholder][i], axis=1)
                if correct_visualized_counter < visualize_correct and corrects_vector[i] == True:
                    visualize.visualize_with_correct(batch[inputs_placeholder][i], predictions[i], true_label,
                                                     name + "_correct")
                    correct_visualized_counter += 1
                elif incorrect_visualized_counter < visualize_incorrect and corrects_vector[i] == False:
                    visualize.visualize_with_correct(batch[inputs_placeholder][i], predictions[i], true_label,
                                                     name + "_incorrect")
                    incorrect_visualized_counter += 1

    precision = correct_num / number_of_examples
    summary = tf.Summary()
    summary.value.add(tag='Accuracy_' + name, simple_value=precision)
    summary_writer.add_summary(summary, learning_step)
    print("Accuracy %.3f" % precision)


if __name__ == '__main__':

    # Load dataset
    train = load_dataset("train.p", False)
    validation = load_dataset("validation.p", False)
    test = load_dataset("test.p", False)

    with tf.Graph().as_default():
        # Wiring
        # model = Sequence()
        # model = SequenceBiggerOutput()
        # model = SequenceReshapedConvolution()
        model = SequenceReshapedConvolutionBatchnorm()
        # model = SequenceReshapedConvolutionDeeper()

        inputs_placeholder, labels_placeholder, keep_prob_placeholder, is_training_placeholder = model.input_placeholders()
        logits = model.inference(inputs_placeholder, keep_prob_placeholder, is_training_placeholder)
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

        # Saver to save checkpoints
        saver = tf.train.Saver(max_to_keep=4)

        # Training
        steps = 30000
        for step in range(steps + 1):
            batch = get_batch(train, inputs_placeholder, labels_placeholder, keep_prob_placeholder, 0.85,
                              is_training_placeholder, True)
            loss_value, summary, _ = session.run([loss, merged_summary, training], feed_dict=batch)
            writer.add_summary(summary, step)
            if step % 1000 == 0:
                print("Step %d, loss %.3f" % (step, loss_value))

                # Save checkpoint
                # TODO if folder exists
                print("Creating checkpoint")
                try:
                    os.makedirs(os.path.join("checkpoints", model.get_name()))
                except:
                    pass
                saver.save(session, os.path.join("checkpoints", model.get_name(), model.get_name()), global_step=step)

                # Visualize at the end of training
                if step == steps:
                    visualize_correct_count = 100
                    visualize_incorrect_count = 100
                    print("Saving visualizations")
                else:
                    visualize_correct_count = 0
                    visualize_incorrect_count = 0

                print("Train accuracy")
                evaluate(train, session, evaluation, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
                         is_training_placeholder, "train", writer, step, visualize_correct_count,
                         visualize_incorrect_count)
                print("Validation accuracy")
                evaluate(validation, session, evaluation, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
                         is_training_placeholder, "validation", writer, step, visualize_correct_count,
                         visualize_incorrect_count)
                print("Test accuracy")
                evaluate(test, session, evaluation, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
                         is_training_placeholder, "test", writer, step, visualize_correct_count,
                         visualize_incorrect_count)
                print()
