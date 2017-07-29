# Sequence of digits recognition
Recognition of sequence of digits using tensorflow. All experiments, descriptions together with images of models and performances are described in [Sequence of digits recognition](http://petr-marek.com/blog/2017/07/05/sequence-digits-recognition/ "Sequence of digits recognition").

![Sequence of digits recognition](http://petr-marek.com/wp-content/uploads/2017/07/Sn%C3%ADmek-obrazovky-68.png)

## How to use
### Single digit recognition
Main file for single digit recognition is [main_single_digit.py](single_digit/main_single_digit.py) in the [single_digit](single_digit) folder. Open the folder, choose model in the main method and run it by:
  
    py -3 main_single_digit.py

### Sequence of digits recognition
You will need to create dataset of sequences of digits. The dataset is created by concatenation of mnist digits. The code for it is in the [prepare_dataset.py](prepare_dataset.py). Run it by:

    py -3 prepare_dataset.py

Three pickle files will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. The file for it is [main_sequence.py](sequence_of_digits/main_sequence.py) in the [sequence_of_digits](sequence_of_digits) folder. Open the folder, choose the model in the main method and run the training by:

    py -3 main_sequence.py

## Models
### Single digit recognition
Single digit recognition uses Mnist dataset from tensorflow.

#### Single layer feedforward model
[feed_forward.py](single_digit/feed_forward.py)

Single layer feedforward model is baseline model.

#### Two layers feedforward model
[feed_forward_two_layers.py](single_digit/feed_forward_two_layers.py)

Two layers feedforward model mainly tests the connection of two layers.

#### Convolutional model
[convolutional.py](single_digit/convolutional.py)

Convolutional model containing three layers of convolution, relu and max pooling, followed by three fully connected layers.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/conv.png" width="200px">
</div>


### Sequence of digits recognition
Sequence of digits recognition uses dataset of digit sequences created by concatenation of digits from Mnist dataset. You can create the dataset as described in the section [How to use](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#how-to-use).

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/Train-e1499360959701.png" width="700px">
</div>

#### Recurrent model
[sequence.py](sequence_of_digits/sequence.py)

Recurrent model uses the same convolutional layers as [Convolutional model](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#convolutional-model). The three fully connected layers are replaced by five times unrolled GRU units followed by single fully connected layer. Plus there is dropout after each convolutional layer which is not shown on the picture. You can disable it by setting keep_prob paramter to 1.0 in the [main_sequence.py](sequence_of_digits/main_sequence.py).

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/sequence-network.png" width="700px">
</div>

#### Recurrent model with bigger output layer
[sequence_bigger_output.py](sequence_of_digits/sequence_bigger_output.py)

It is the improvement of the [Recurrent model](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#recurrent-model). The single fully connected output layer is replaced by three fully connected layers and relu activation functions. There is dropout between the first and the second, and second and the third. You can tune ropout by keep_prob parameter int he [main_sequence.py](sequence_of_digits/main_sequence.py). It achieves slightly better accuracy then the [Recurrent model](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#recurrent-model).

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/Recurrence_bigger_output_dropout.png" width="700px">
</div>

#### Recurrent model with reshaped convolution
[sequence_reshaped_convolution.py](sequence_of_digits/sequence_reshaped_convolution.py)

It is model with same layers as [sequence.py](sequence_of_digits/sequence.py). The change is in the 
size of convolutions and number of their filters. The size of convolution is decreasing and the number of filters are increasing in the layers.

#### Recurrent model with reshaped convolution and batch normalization
[sequence_reshaped_convolution_batchnorm.py](sequence_of_digits/sequence_reshaped_convolution_batchnorm.py)

It is model similar to [sequence_reshaped_convolution.py](sequence_of_digits/sequence_reshaped_convolution.py) and it adds batch normalization layers. This model achieves the best accuracy on the testing set.
The convolutional part of the model is shown bellow.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/Batchnorm_model.png" width="700px">
</div>

#### Recurrent model with deeper convolution
[sequence_reshaped_convolution_deeper.py](sequence_of_digits/sequence_reshaped_convolution_deeper.py)

It is model adding the fourth convolutional layer to [sequence_reshaped_convolution.py](sequence_of_digits/sequence_reshaped_convolution.py).
The convolutional part of the model is shown bellow.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/deeper_architecture.png" width="300px">
</div>

#### Recurrent model with doubled layers
[sequence_reshaped_convolution_batchnorm_double.py](sequence_of_digits/sequence_reshaped_convolution_batchnorm_double.py)

It is model doubling the layers of the [sequence_reshaped_convolution_batchnorm.py](sequence_of_digits/sequence_reshaped_convolution_batchnorm.py). 
The convolutional part of the model is shown bellow.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/double-architecture-2.png" width="250px">
</div>
 
#### Recurrent model with overlapping stride
[sequence_reshaped_convolution_batchnorm_stride.py](sequence_of_digits/sequence_reshaped_convolution_batchnorm_stride.py)

It is the same model as [sequence_reshaped_convolution_batchnorm.py](sequence_of_digits/sequence_reshaped_convolution_batchnorm.py) except the stride of max pooling layers.
The max pooled regions are overlapped.