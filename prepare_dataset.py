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
    train_dataset = []
    print("Done")

    print("Creating validation dataset")
    validation_dataset = make_dataset(mnist.validation, 15000, 5)
    pickle.dump(validation_dataset, open("validation.p", "wb"))
    validation_dataset = []
    print("Done")

    print("Creating testing dataset")
    test_dataset = make_dataset(mnist.test, 30000, 5)
    pickle.dump(test_dataset, open("test.p", "wb"))
    test_dataset = []
    print("Done")

    load_dataset("train.p", True)
    load_dataset("validation.p", True)
    load_dataset("test.p", True)
