from random import randrange

from matplotlib import patches
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import pickle

'''
Creates dataset of digit sequences by concatenating of mnist digits

Params:
    dataset - dataset to use (mnist.train, mnist.validation or mnist.test)
    num_examples - how many examples to create (size of created dataset)
    length - the length of digit sequences
    debug - show generated examples
'''


def make_dataset(dataset, num_examples, length, debug=False):
    examples = []
    labels = []
    permutation = np.random.permutation(dataset.num_examples)
    permutation_index = 0
    for i in range(num_examples):
        example = []
        label = []
        for j in range(length):
            number = np.reshape(dataset.images[permutation_index], [28, 28])
            if len(example) == 0:
                example = number
                label = dataset.labels[permutation_index]
            else:
                example = np.append(example, number, axis=1)
                label = np.vstack([label, dataset.labels[permutation_index]])
            permutation_index += 1
            if permutation_index >= len(permutation):
                permutation = np.random.permutation(dataset.num_examples)
                permutation_index = 0
        if debug:
            print(label)
            plt.imshow(example, cmap='gray')
            plt.show()
        examples.append(example)
        labels.append(label)
    return {"examples": examples, "labels": labels}


'''
Makes dataset for localization

Params:
    dataset - dataset of sequences to use (created by function make_dataset())
    height - height of new images
    width - width of new images
    debug - show generated examples
'''


def make_localization_dataset(dataset, pickle_file_name, height, width, debug=False):
    f = open(pickle_file_name, 'wb')
    for i in range(len(dataset["examples"])):
        example = dataset["examples"][i]
        label = dataset["labels"][i]
        new_example = np.zeros([height, width])
        max_h = height - example.shape[0]
        max_w = width - example.shape[1]
        h_transition = randrange(max_h)
        w_transition = randrange(max_w)
        x = w_transition + int(example.shape[1] / 2)
        y = h_transition + int(example.shape[0] / 2)
        h = example.shape[0]
        w = example.shape[1]

        new_example[h_transition:h_transition + example.shape[0],
        w_transition:w_transition + example.shape[1]] = example
        position = [x, y, h, w]
        pickle.dump({"example": new_example, "label": label, "position": position}, f)

        if debug:
            print(position)
            fig, ax = plt.subplots(1)
            ax.imshow(new_example, cmap='gray')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)
            plt.show()
    f.close()

'''
Makes dataset for localization with variable length of senquences

Params:
    dataset - dataset of sequences to use (Mnist.train, Mnist.validation or Mnist.test)
    pickle_file_name - name of file in which the newly created dataset will be saved
    num_examples - number of created examples
    max_length - maximal length of sequence
    height - height of new images
    width - width of new images
    debug - show generated examples
'''

def make_variable_length_dataset(dataset, pickle_file_name, num_examples, max_length, height, width, debug=False):
    f = open(pickle_file_name, 'wb')
    permutation = np.random.permutation(dataset.num_examples)
    permutation_index = 0
    for i in range(num_examples):
        example = []
        label = []
        length = randrange(1, max_length + 1)
        for j in range(max_length):
            number = np.reshape(dataset.images[permutation_index], [28, 28])
            if j < length:
                if len(example) == 0:
                    example = number
                    label = np.append(dataset.labels[permutation_index], [0])
                else:
                    example = np.append(example, number, axis=1)
                    label = np.vstack([label, np.append(dataset.labels[permutation_index], [0])])
            else:
                label = np.vstack([label, np.append(np.zeros([10]), [1])])
            permutation_index += 1
            if permutation_index >= len(permutation):
                permutation = np.random.permutation(dataset.num_examples)
                permutation_index = 0
        if debug:
            print(label)
            plt.imshow(example, cmap='gray')
            plt.show()

        new_example = np.zeros([height, width])
        max_h = height - example.shape[0]
        max_w = width - example.shape[1]
        h_transition = randrange(max_h)
        w_transition = randrange(max_w)
        x = w_transition + int(example.shape[1] / 2)
        y = h_transition + int(example.shape[0] / 2)
        h = example.shape[0]
        w = example.shape[1]

        new_example[h_transition:h_transition + example.shape[0],
        w_transition:w_transition + example.shape[1]] = example
        position = [x, y, h, w]
        pickle.dump({"example": new_example, "label": label, "position": position}, f)

        if debug:
            print(label)
            print(position)
            fig, ax = plt.subplots(1)
            ax.imshow(new_example, cmap='gray')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)
            plt.show()
            print()
    f.close()


'''
Loads dataset from pickle file

Params:
    file_name - name of pickle file
    debug - show generated examples
'''


def load_dataset(file_name, debug=False):
    dataset = pickle.load(open(file_name, "rb"))
    if debug:
        for i in range(len(dataset["examples"])):
            print(dataset["labels"][i])
            plt.imshow(dataset["examples"][i], cmap='gray')
            plt.show()
    return dataset


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    print("Creating testing dataset")
    train_dataset = make_dataset(mnist.train, 165000, 5)
    pickle.dump(train_dataset, open("train.p", "wb"))
    print("Creating testing dataset for localization")
    make_localization_dataset(train_dataset, "train_localization.p", 128, 256, False)
    train_dataset = []
    print("Done")

    print("Creating validation dataset")
    validation_dataset = make_dataset(mnist.validation, 15000, 5)
    pickle.dump(validation_dataset, open("validation.p", "wb"))
    print("Creating validation dataset for localization")
    make_localization_dataset(validation_dataset, "validation_localization.p", 128, 256, False)
    validation_dataset = []
    print("Done")

    print("Creating testing dataset")
    test_dataset = make_dataset(mnist.test, 30000, 5)
    pickle.dump(test_dataset, open("test.p", "wb"))
    print("Creating testing dataset for localization")
    make_localization_dataset(test_dataset, "test_localization.p", 128, 256, False)
    test_dataset = []
    print("Done")

    print("Creating variable length localization training dataset")
    make_variable_length_dataset(mnist.train, "train_variable_localization.p", 165000, 5, 128, 256, debug=False)
    print("Done")

    print("Creating variable length localization validation dataset")
    make_variable_length_dataset(mnist.validation, "validation_variable_localization.p", 15000, 5, 128, 256,
                                 debug=False)
    print("Done")

    print("Creating variable length localization testing dataset")
    make_variable_length_dataset(mnist.validation, "test_variable_localization.p", 30000, 5, 128, 256, debug=False)
    print("Done")
