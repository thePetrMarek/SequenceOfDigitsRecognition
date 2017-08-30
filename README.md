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

Three pickle files (train.p, validation.p, test.p) will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. Main file for it is [main_sequence.py](sequence_of_digits/main_sequence.py) in the [sequence_of_digits](sequence_of_digits) folder. Open the folder, choose the model in the main method and run the training by:

    py -3 main_sequence.py

### Sequence of digits recognition and localization
You will need to create dataset of sequences of digits. The dataset is created by concatenation of mnist digits. The code for it is in the [prepare_dataset.py](prepare_dataset.py). Run it by:

    py -3 prepare_dataset.py
    
Three pickle files (train_localization.p, validation_localization.p, test_localization.p) will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. Main file for it is [main_sequence_localization.py](sequence_of_digits_localization/main_sequence_localization.py) in the [sequence_of_digits_localization](sequence_of_digits_localization) folder. Open the folder, choose the model in the main method and run the training by:

    py -3 main_sequence_localization.py

### Sequences of digits with variable lenght
You will need to create dataset of sequences of digits. The dataset is created by concatenation of mnist digits. The code for it is in the [prepare_dataset.py](prepare_dataset.py). Run it by:

    py -3 prepare_dataset.py
    
Three pickle files (train_variable_localization.p, validation_variable_localization.p, test_variable_localization.p) will be created containing training, validation and testing datasets.

The next step is to run main file for training of the sequence recognition models. Main file for it is [main_sequences_variable_length.py](sequences_of_variable_length/main_sequences_variable_length.py) in the [sequences_of_variable_length](sequences_of_variable_length) folder. Open the file, choose the model in the main method and run the training by:

    py -3 main_sequence_variable_length.py

### SVHN recognition
This task uses real images of house numbers from Street View. Download dataset from [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/) and place it into project's root. Run [prepare_svhn_dataset.py](prepare_svhn_dataset.py) by:
    
    py -3 prepare_svhn_dataset.py
    
This will create folder ``SVHN_data`` containing prepared dataset. Use file [SVHN_recognition/main_SVHN_recognition.py](SVHN_recognition/main_SVHN_recognition.py) for training in the [SVHN_recognition](SVHN_recognition) folder. Open the file, choose the model in the main method and run the training by:

    py -3 main_SVHN_recognition.py

## Models
### Single digit recognition
Single digit recognition uses Mnist dataset from tensorflow.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/decompressed.jpg" width="200px">
</div>

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

### Sequence of digits recognition and localization
Sequence of digits recognition and localization uses dataset of digit sequences created by concatenation of digits from Mnist dataset. The digit sequences are placed on the random location of canvas. You can create the dataset as described in the section [How to use](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#how-to-use).

The task is to correctly classify the sequence of numbers and to localize it. The location of sequence is defened by x, y coordinates and width and height of bounding box.

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/loc-e1501527743191.png" width="700px">
</div>

#### Deep localization model with weighted loss
[deep_localization_weighted_loss.py](sequence_of_digits_localization/deep_localization_weighted_loss.py)

Model able to learn the classification of sequences of digits and their localization. The loss function is

    loss = 1000 * “classification loss (cross entropy)” + “localization error (meaned squared error)”
    
<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/graph-runweightLoss.png" width="600px">
</div>

### Sequences of variable lengths
This is generalization of [Sequence of digits recognition and localization](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#sequence-of-digits-recognition-and-localization). 
The task is to classify and to localize sequence of digits again, but the sequence has variable length this time. The maximal size of sequence is known. 
You can create the dataset as described in the section [How to use](https://github.com/thePetrMarek/SequenceOfDigitsRecognition#how-to-use).

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/variable_length_dataset-e1501790547739.png" width="700px">
</div>

#### Deep localization model with weighted loss for variable length
[deep_localization_weighted_loss_variable_length.py](sequences_of_variable_length/deep_localization_weighted_loss_variable_length.py)

This is the same model as [deep_localization_weighted_loss.py](sequence_of_digits_localization/deep_localization_weighted_loss.py). Only difference is the adaptation to 
output special "no digit" character.

#### Deeper localization model with weighted loss for vasibale length
[deep_localization_weighted_loss_variable_length_deeper.py](sequences_of_variable_length/deep_localization_weighted_loss_variable_length_deeper.py)

This model is almost the same as [deep_localization_weighted_loss_variable_length.py](sequences_of_variable_length/deep_localization_weighted_loss_variable_length.py). Only difference is sixth convolutional layer.

### SVHN recognition
This task is to recognize house numbers in real word images taken from Google Street View. 

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/2-8-8-10-10-10.png" width="700px">
</div>

#### Deep localization weighted loss variable length
[deep_localization_weighted_loss_variable_length.py](SVHN_recognition/deep_localization_weighted_loss_variable_length_deep.py)

The same model as [deep_localization_weighted_loss_variable_length.py](sequences_of_variable_length/deep_localization_weighted_loss_variable_length.py), only difference is that it can output six digits instead of five.

#### Paper convolution
[svhn_paper_convolution.py](SVHN_recognition/svhn_paper_convolution.py)

This model is inspired by paper Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks. It also doesn't use reccurent recognition head. It uses six fully connected layers instead. Each for one digit.

#### Paper convolution with dropout
[svhn_paper_convolution_dropout_output.py](SVHN_recognition/svhn_paper_convolution_dropout_output.py)

This model adds dropout before output layers of model [svhn_paper_convolution.py](SVHN_recognition/svhn_paper_convolution.py).

<div align="center">
  <img src="http://petr-marek.com/wp-content/uploads/2017/07/graph-runSVHN_paper_convolution_dropout_output_0.85C.png" width="800px">
</div>

#### No maxpooling
[svhn_transfer_learning_no_maxpool.py](SVHN_recognition/svhn_transfer_learning_no_maxpool.py)

This model replaces 2x2 max pooling layers by 2x2 convolutional layers with 2x2 stride.
