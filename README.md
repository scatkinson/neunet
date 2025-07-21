# neunet
Neural Network from Scratch with Python and Numpy

## [0. Contents](#0._Contents) <a id='0._Contents'></a>

## [1. Code Base](#1._Code_Base)

* ### [1.0. `bin`](#1.0._bin)
* ### [1.1. `data`](#1.1._data)
* ### [1.2. `neunet`](#1.2._neunet)
  * #### [1.2.0. `activations.py`](#1.2.0._activations.py)
  * #### [1.2.1. `config.py`](#1.2.1._config.py)
  * #### [1.2.2. `configamend.py`](#1.2.2_configamend.py)
  * #### [1.2.3. `constants.py`](#1.2.3._constants.py)
  * #### [1.2.4. `layer.py`](#1.2.4._layer.py)
  * #### [1.2.5. `log_wu.py`](#1.2.5._log_wu.py)
  * #### [1.2.6. `losses.py`](#1.2.6._losses.py)
  * #### [1.2.7. `network.py`](#1.2.7._network.py)
  * #### [1.2.8. `trainer.py`](#1.2.8._trainer.py)
  * #### [1.2.9. `trainer_config.py`](#1.2.9._trainer_config.py)
  * #### [1.2.10. `util.py`](#1.2.10._util.py)

## 1. Code Base <a id='1._Code_Base'></a>

### 1.0. `bin` <a id='1.0._bin'></a>

Each script in the `bin` directory is an end-user Python scripts to be executed along with the `conf` subdirectory corresponding config files.
Here is an example of how to run one of these scripts (from the root repo directory):

`python bin/classifier_trainer.py -c bin/conf/classifier_test_config.yml`

### 1.1. `data` <a id='1.1._data'></a>

The `data` directory contains the data for training and evaluating the NN models.

### 1.2. `neunet` <a id='1.2._neunet'></a>

The `neunet` directory is a Python library containing all the classes for assembling, training, and evaluating the available neural networks.

#### 1.2.0. `activations.py` <a id='1.2.0._activations.py'></a>

The `activations.py` module contains the base `Activation` class and the available activation functions as child classes. 
Each class contains the function and its gradient.
The available activation functions include:
* `Identity`: $f(x) = x$
* `ReLu`: $f(x) = \text{max}(x,0)$
* `SoftMax`: 
```math
f(\left\langle x_1, \dots, x_n\right\rangle) = \left\langle\frac{e^{x_1}}{\sum_{j=1}^n e^{x_j}}, \dots, \frac{e^{x_n}}{\sum_{j=1}^n e^{x_j}}\right\rangle
```
* `Sigmoid`: $f(x) = \frac{1}{1+e^{-x}}$
* `HardTanh`: $f(x) = \text{min}(1, \text{max}(-1,x))$

#### 1.2.1. `config.py` <a id='1.2.1._config'></a>

The `config.py` module contains the base config class that is used to pass config parameters to relevant classes.

#### 1.2.2. `configamend.py` <a id='1.2.2_configamend'></a>

The `configamend.py` module contains a few functions to be used to ensure the parameters passes from the config file are in the proper format.

#### 1.2.3. `constants.py` <a id='1.2.3._constants.py'></a>

The `constants.py` module contains the global vairables (dict keys, col names, etc.) that are used throughout the library.

#### 1.2.4. `layer.py` <a id='1.2.4._layer.py'></a>

The `layer.py` module contains the bases `Layer` class on which the following layers classes are built:
* `DenseLayer`: a fully connected layer.
* `ConvolutionalLayer`: a convolutional layer for convolutional neural networks.
* `MaxPool`: a max pool layer to further filter the output of the convolutional layer.

#### 1.2.5. `log_wu.py` <a id='1.2.5._log_wu.py'></a>

The `log_wu.py` module contains helper functions for managing the file logging of the library.

#### 1.2.6. `losses.py` <a id='1.2.6._losses.py'></a>

The `losses.py` module contains the base `Loss` class on which the specific loss classes are built.
Each loss class contains the loss function and its gradient.
The following loss functions are enabled:
* `CrossEntropy`: 
```math
f(\left\langle y_1,\dots, y_n\right\rangle, \left\langle \hat{y_1}, \dots, \hat{y_n}\right\rangle) = -\frac{1}{n}\cdot \sum_{j=1}^n y_j\log(\hat{y_j})
```
* `MSE`:
```math
f(\left\langle y_1,\dots, y_n\right\rangle, \left\langle \hat{y_1}, \dots, \hat{y_n}\right\rangle) = \frac{1}{n} \cdot \sum_{j=1}^n (y_j - \hat{y_j})^2
```
* `RMSE`:
```math
f(\left\langle y_1,\dots, y_n\right\rangle, \left\langle \hat{y_1}, \dots, \hat{y_n}\right\rangle) = \sqrt{\frac{1}{n} \cdot \sum_{j=1}^n (y_j - \hat{y_j})^2}
```
* `LogLoss`:
```math
f(y,\hat{y}) = -(y\log(\hat{y}) + (1-y)\log(1-\hat{y})
```
#### 1.2.7. `network.py` <a id='1.2.7._network.py'></a>
 
The `network.py` module contains the `Network` class which manages the neural network.

* `add`: method for appending a layer to the network.
* `compile`: obtains the input shape, output shape, and activation function for each layer.
* `init_weights`: initializes the weights for each layer of the network.
* `forward`: performs the forward pass of the network
* `backprop`: performs backpropagation of the network to obtain the gradients at each layer.
* `update`: updates the weights and biases of each layer using the gradients and learning rate.
* `train`: performs network training
* `batch_train`: performs network training on sub-batches of the input data

#### 1.2.8. `trainer.py` <a id=`1.2.8._trainer.py'></a>

The `trainer.py` module contains the base `Trainer` class for training neural networks. 
This class orchestrates the training and evaluation of the neural network.  
Here are the child classes of the `Trainer` class:
* `BinaryClassifierTrainer`: orchestrates training for a binary classification model
* `CNNTrainer`: orchestrates training for a convolutional neural network classification model
* `RegressionTrainer`: orchestrates trainings for a regression model.

#### 1.2.9. `trainer_config.py` <a id='1.2.9._trainer_config.py'></a>

The `trainer_config.py` module contains the `TrainerConfig` class for passing relevant parameters to a `Trainer` class.

#### 1.2.10. `util.py` <a id='1.2.10._util.py'></a>

The `util.py` module contains miscellaneous utility functions used in the library.