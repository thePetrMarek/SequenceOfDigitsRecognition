# Sequence of digits recognition
Recognition of sequence of digits using tensorflow. All experiments are described on [Sequence of digits recognition](http://petr-marek.com/blog/2017/07/05/sequence-digits-recognition/ "Sequence of digits recognition").

## How to use
### Single digit recognition
Main file for single digit recognition is [main_single_digit.py](main_single_digit.py). Choose model in the main method and run it by:
  
    py -3 main_single_digit.py

### Sequence of digits recognition
You will need to create dataset of sequences of digits. The dataset is created by concatenation of mnist digits. The code for it is in the [prepare_dataset.py](prepare_dataset.py). Run it by:

    py -3 prepare_dataset.py

Three pickle files will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. The file for it is [main_single_digit.py](main_sequence.py). Run the training by:

    py -3 main_single_digit.py

## Models
### Single digit recognition
Single digit recognition uses Mnist dataset from tensorflow.
