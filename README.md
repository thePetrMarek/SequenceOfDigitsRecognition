# Sequence of digits recognition
Recognition of sequence of digits using tensorflow. All experiments, descriptions together with images of models and performances are described in [Sequence of digits recognition](http://petr-marek.com/blog/2017/07/05/sequence-digits-recognition/ "Sequence of digits recognition").

## How to use
### Single digit recognition
Main file for single digit recognition is [main_single_digit.py](main_single_digit.py). Choose model in the main method and run it by:
  
    py -3 main_single_digit.py

### Sequence of digits recognition
You will need to create dataset of sequences of digits. The dataset is created by concatenation of mnist digits. The code for it is in the [prepare_dataset.py](prepare_dataset.py). Run it by:

    py -3 prepare_dataset.py

Three pickle files will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. The file for it is [main_sequence.py](main_sequence.py). Run the training by:

    py -3 main_sequence.py

## Models
### Single digit recognition
Single digit recognition uses Mnist dataset from tensorflow.

#### Single layer feedforward model
[feed_forward.py](feed_forward.py)
Single layer feedforward model is baseline model.

#### Two layers feedforward model
[feed_forward_two_layers.py](feed_forward_two_layers.py)
Two layers feedforward model mainly tests the connection of two layers.

#### Convolutional model
[convolutional.py](convolutional.py)
Convolutional model containing three layers of convolution, relu and max pooling, followed by three fully connected layers.

### Sequence of digits recognition
Sequence of digits recognition uses dataset of digit sequences created by concatenation of digits from Mnist dataset. You can create the dataset as described in the section [How to use](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#how-to-use).

#### Recurrent model
Recurrent model uses the same convolutional layers as [Convolutional model](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#convolutional-model). The three fully connected layers are replaced by five times unrolled GRU units followed by single fully connected layer.
