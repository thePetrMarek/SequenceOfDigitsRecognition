import os

import numpy as np
import tensorflow as tf

from SVHN_dataset import SVHNDataset
from SVHN_recognition.deep_localization_weighted_loss_variable_length import DeepLocalizationWeightedLossVariableLength
from SVHN_recognition.svhn_paper_convolution import SVHNPaperConvolution
from SVHN_recognition.svhn_paper_convolution_dropout_output import SVHNPaperConvolutionDropoutOutput
from SVHN_recognition.svhn_transfer_learning import SVHNTransferLearning
from SVHN_recognition.svhn_transfer_learning_no_maxpool import SVHNNoMaxpool
from visualize import Visualize
import matplotlib.pyplot as plt
import pickle


def get_batch(dataset, inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder,
              keep_prob_placeholder_conv,
              keep_prob_val, keep_prob_conv_val,
              is_training_placeholder, is_traininig):
    batch = dataset.load(50)
    inputs = batch['examples']
    labels = batch['labels']
    positions = batch['positions']

    return {"batch": {inputs_placeholder: inputs, labels_placeholder: labels, positions_placeholder: positions,
                      keep_prob_placeholder: keep_prob_val,
                      keep_prob_placeholder_conv: keep_prob_conv_val,
                      is_training_placeholder: is_traininig}, "end_of_file": batch["end_of_file"]}


def evaluate(dataset, session, operation, inputs_placeholder, labels_placeholder, positions_placeholder,
             keep_prob_placeholder, keep_prob_placeholder_conv,
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
                                 keep_prob_placeholder, keep_prob_placeholder_conv, 1, 1,
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
    train_localization = SVHNDataset("../SVHN_data/extratrain/", "extratrain.json", np.zeros((128, 256, 3)),
                                     np.ones((128, 256, 3)))
    all_training_example = False
    sum = None
    number_of_examples = 0
    while not all_training_example:
        loaded = train_localization.load(500)
        all_training_example = loaded["end_of_file"]
        examples = loaded["examples"]
        if sum is None:
            sum = np.zeros((128, 256, 3))
        batch_sum = np.sum(examples, axis=0)
        sum = np.sum([sum, batch_sum], axis=0)
        number_of_examples += len(examples)
        print("Calculated mean of " + str(number_of_examples) + " examples")
    mean = sum / number_of_examples

    train_localization = SVHNDataset("../SVHN_data/extratrain/", "extratrain.json", np.zeros((128, 256, 3)),
                                     np.ones((128, 256, 3)))
    all_training_example = False
    squares = None
    number_of_calculated = 0
    while not all_training_example:
        loaded = train_localization.load(500)
        all_training_example = loaded["end_of_file"]
        examples = loaded["examples"]
        if squares is None:
            squares = np.zeros((128, 256, 3))
        sub = examples - mean
        squares += np.sum(np.power(sub, 2), axis=0)
        number_of_calculated += len(examples)
        print("Calculated std of " + str(number_of_calculated) + " examples")
    std = np.sqrt(squares / number_of_examples)
    plt.imshow(np.around(mean))

    plt.imshow(np.around(std))
    return mean, std


if __name__ == '__main__':
    # Use it at first to calculate mean and std of training dataset, use created pickle file later
    mean, std = calculate_normalization_parameters()
    pickle.dump({"mean": mean, "std": std}, open("mean_std.p", "wb"))
    # mean_std = pickle.load(open("mean_std.p", "rb"))
    # mean = mean_std["mean"]
    # std = mean_std["std"]

    # Load dataset
    train_localization = SVHNDataset("../SVHN_data/extratrain/", "extratrain.json", mean, std)

    with tf.Graph().as_default():
        # Wiring
        # model = DeepLocalizationWeightedLossVariableLength()
        # model = SVHNPaperConvolution()
        # model = SVHNTransferLearning()
        model = SVHNNoMaxpool()

        inputs_placeholder, labels_placeholder, positions_placeholder, keep_prob_placeholder, keep_prob_placeholder_conv, is_training_placeholder = model.input_placeholders()
        logits, predicted_positions, regularization = model.inference(inputs_placeholder, keep_prob_placeholder,
                                                                      keep_prob_placeholder_conv,
                                                                      is_training_placeholder)
        loss = model.loss(logits, labels_placeholder, predicted_positions, positions_placeholder, regularization)
        training = model.training(loss["total_loss"], 0.0001)
        evaluation = model.evaluation(logits, labels_placeholder, predicted_positions, positions_placeholder)

        # Initialization
        session = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        session.run(init)

        # visualize graph
        writer = tf.summary.FileWriter("visualizations/" + model.get_name())
        writer.add_graph(session.graph)

        loader = tf.train.Saver()

        # Summaries
        merged_summary = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=4)

        RESTORE = False
        ckpt = "checkpoints/SVHN_no_maxpool_reweighting/SVHN_no_maxpool_reweighting-0"
        continue_from_step = 0

        if RESTORE:
            loader.restore(session, ckpt)
            start_step = continue_from_step
        else:
            start_step = 0

        # Training
        steps = 30000
        for step in range(start_step, steps + 1):
            batch = get_batch(train_localization, inputs_placeholder, labels_placeholder, positions_placeholder,
                              keep_prob_placeholder, keep_prob_placeholder_conv, 0.5, 0.9,
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
                train_localization_evaluation = SVHNDataset("../SVHN_data/extratrain/", "extratrain.json", mean, std)
                evaluate(train_localization_evaluation, session, evaluation, inputs_placeholder, labels_placeholder,
                         positions_placeholder,
                         keep_prob_placeholder,
                         keep_prob_placeholder_conv,
                         is_training_placeholder, model.get_name(), "train", writer, step,
                         visualize_correct_count,
                         visualize_incorrect_count)
                print("Test accuracy")
                test_localization_evaluation = SVHNDataset("../SVHN_data/test/new/", "test.json", mean, std)
                evaluate(test_localization_evaluation, session, evaluation, inputs_placeholder, labels_placeholder,
                         positions_placeholder,
                         keep_prob_placeholder,
                         keep_prob_placeholder_conv,
                         is_training_placeholder, model.get_name(), "test", writer, step,
                         visualize_correct_count,
                         visualize_incorrect_count)
                print()
