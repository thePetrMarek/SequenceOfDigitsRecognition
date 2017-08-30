import os

import numpy as np
import tensorflow as tf

from localization_dataset import LocalizationDataset
from sequences_of_variable_length.deep_localization_weighted_loss_variable_length_deeper import \
    DeepLocalizationWeightedLossVariableLengthDeeper
from visualize import Visualize
import matplotlib.pyplot as plt


def get_batch(dataset, inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder,
              keep_prob_val,
              is_training_placeholder, is_traininig):
    batch = dataset.load(50)
    inputs = batch['examples']
    labels = batch['labels']
    positions = batch['positions']

    return {"batch": {inputs_placeholder: inputs, labels_placeholder: labels, positions_placeholder: positions,
                      keep_prob_placeholder: keep_prob_val,
                      is_training_placeholder: is_traininig}, "end_of_file": batch["end_of_file"]}


def evaluate(dataset, session, operation, inputs_placeholder, labels_placeholder, positions_placeholder,
             keep_prob_placeholder,
             is_training_placeholder, model_name, name,
             summary_writer, learning_step, visualize_correct=0, visualize_incorrect=0):
    visualize = Visualize()
    correct_visualized_counter = 0
    incorrect_visualized_counter = 0

    correct_num = 0
    total_position_error = 0
    number_of_examples = 0
    number_of_characters = 0
    correct_num_characters = 0
    while True:
        batch_object = get_batch(dataset, inputs_placeholder, labels_placeholder, positions_placeholder,
                                 keep_prob_placeholder, 1,
                                 is_training_placeholder, False)
        if batch_object["end_of_file"]:
            break
        batch = batch_object["batch"]
        number_of_examples += len(batch[inputs_placeholder])
        corrects_in_batch, corrects_vector, predictions, batch_position_error, predicted_positions, total_correct_characters, total_characters = session.run(
            operation, feed_dict=batch)
        correct_num += corrects_in_batch
        total_position_error += batch_position_error
        number_of_characters += total_characters
        correct_num_characters += total_correct_characters

        # visualize correct and incorrect recognitions
        if incorrect_visualized_counter < visualize_incorrect or correct_visualized_counter < visualize_correct:
            for i in range(len(batch[inputs_placeholder])):
                true_label = np.argmax(batch[labels_placeholder][i], axis=1)
                if correct_visualized_counter < visualize_correct and corrects_vector[i] == True:
                    visualize.visualize_with_correct_label_position(batch[inputs_placeholder][i], predictions[i],
                                                                    true_label,
                                                                    predicted_positions[i],
                                                                    batch[positions_placeholder][i],
                                                                    os.path.join(model_name, name) + "_correct")
                    correct_visualized_counter += 1
                elif incorrect_visualized_counter < visualize_incorrect and corrects_vector[i] == False:
                    visualize.visualize_with_correct_label_position(batch[inputs_placeholder][i], predictions[i],
                                                                    true_label,
                                                                    predicted_positions[i],
                                                                    batch[positions_placeholder][i],
                                                                    os.path.join(model_name, name) + "_incorrect")
                    incorrect_visualized_counter += 1

    sequence_accuracy = correct_num / number_of_examples
    character_accuracy = correct_num_characters / number_of_characters
    position_error = total_position_error / (number_of_examples / 50)

    summary = tf.Summary()
    summary.value.add(tag='Sequence_accuracy_' + name, simple_value=sequence_accuracy)
    summary.value.add(tag='Character_accuracy_' + name, simple_value=character_accuracy)
    summary.value.add(tag='Position_error_' + name, simple_value=position_error)
    summary_writer.add_summary(summary, learning_step)

    print("Number of correct examples: " + str(correct_num) + "/" + str(number_of_examples))
    print("Number of correct characters: " + str(correct_num_characters) + "/" + str(number_of_characters))

    print("Sequence accuracy %.3f" % sequence_accuracy)
    print("Character accuracy %.3f" % character_accuracy)
    print("Position error %.3f" % position_error)
    print()


def calculate_normalization_parameters():
    train_localization = LocalizationDataset("../train_variable_localization.p")
    all_training_example = False
    sum = None
    number_of_examples = 0
    while not all_training_example:
        loaded = train_localization.load(50)
        all_training_example = loaded["end_of_file"]
        examples = loaded["examples"]
        for example in examples:
            if sum is None:
                sum = example
            else:
                sum += example
        number_of_examples += len(examples)
    mean = sum / number_of_examples

    train_localization = LocalizationDataset("../train_variable_localization.p")
    all_training_example = False
    squares = None
    while not all_training_example:
        loaded = train_localization.load(50)
        all_training_example = loaded["end_of_file"]
        examples = loaded["examples"]
        for example in examples:
            if squares is None:
                squares = np.power(example - mean, 2)
            else:
                squares += np.power(example - mean, 2)
    std = np.sqrt(squares / number_of_examples)
    plt.imshow(mean, cmap='gray')
    plt.show()

    plt.imshow(std, cmap='gray')
    plt.show()
    return mean, std


if __name__ == '__main__':
    # Load dataset
    train_localization = LocalizationDataset("../train_variable_localization.p")

    with tf.Graph().as_default():
        # Wiring
        # model = DeepLocalizationWeightedLossVariableLength()
        model = DeepLocalizationWeightedLossVariableLengthDeeper()

        inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder, is_training_placeholder = model.input_placeholders()
        logits, predicted_positions = model.inference(inputs_placeholder, keep_prob_placeholder,
                                                      is_training_placeholder)
        loss = model.loss(logits, labels_placeholder, predicted_positions, positions_placeholder)
        training = model.training(loss["total_loss"], 0.0001)
        evaluation = model.evaluation(logits, labels_placeholder, predicted_positions, positions_placeholder)

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
            batch = get_batch(train_localization, inputs_placeholder, labels_placeholder, positions_placeholder,
                              keep_prob_placeholder, 0.8,
                              is_training_placeholder, True)["batch"]
            loss_value, summary, _ = session.run([loss, merged_summary, training], feed_dict=batch)
            writer.add_summary(summary, step)
            if step % 1000 == 0:
                print("Step %d, Total loss %.3f, Character loss %.3f, Position loss %.3f" % (
                    step, loss_value["total_loss"], loss_value["logits_loss"], loss_value["positions_loss"]))
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
                train_localization_evaluation = LocalizationDataset("../train_variable_localization.p")
                evaluate(train_localization_evaluation, session, evaluation, inputs_placeholder, labels_placeholder,
                         positions_placeholder,
                         keep_prob_placeholder,
                         is_training_placeholder, model.get_name(), "train", writer, step,
                         visualize_correct_count,
                         visualize_incorrect_count)
                print("Validation accuracy")
                validation_localization_evaluation = LocalizationDataset("../validation_variable_localization.p")
                evaluate(validation_localization_evaluation, session, evaluation, inputs_placeholder,
                         labels_placeholder, positions_placeholder,
                         keep_prob_placeholder,
                         is_training_placeholder, model.get_name(), "validation", writer, step,
                         visualize_correct_count,
                         visualize_incorrect_count)
                print("Test accuracy")
                test_localization_evaluation = LocalizationDataset("../test_variable_localization.p")
                evaluate(test_localization_evaluation, session, evaluation, inputs_placeholder, labels_placeholder,
                         positions_placeholder,
                         keep_prob_placeholder,
                         is_training_placeholder, model.get_name(), "test", writer, step,
                         visualize_correct_count,
                         visualize_incorrect_count)
                print()
