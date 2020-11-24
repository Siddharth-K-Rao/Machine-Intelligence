## Implementation:
    1) The neural network designed is a simple densely connected neural network.
    2) The network consists of a 3-layer network (2-hidden layers, output layer).
    3) Layer weights are initialized using Xavier initialization.
    4) Activation functions of 1st hidden layer, 2nd hidden layer and output layer are reLu, reLu and sigmoid respectively.
    5) The network uses recursive backpropagation for weights and bias corrections.

## Hyperparameters:
    1) number of neurons per layer = [128, 32, 1]
    2) epochs = 500
    3) Learning rate = 0.0005
    4) Optimizer choice = 'Adam
    5) Adam optimizer parameters beta1 = 0.9, beta2 = 0.999, epsilon = 10^-8

## Key features of design:</br>
    1) The network is adaptable to change in number of neurons per layer example - [128, 128, 1], [256, 128, 1]
    2) Layer weights are initialized using Xavier method so that the neuron activation functions are not starting out in saturated or dead regions. Xavier method helps to initialize weights with random values that are not "too small" and not "too large".
    3) Adam optimizer is used for achieving good results fast.

## Implementations beyond basics:
    1) Choice of weight initializer is not limited to "Xavier". We have added "plain", "he" in addition.
    2) Gradient descent optimizer is also provided in addition to Adam.
    3) Number of neurons per layer can be tweaked accordingly.
    These additional features helps to build models that work with datasets beyond the LBW dataset.

## Steps to run files:
    1) Data pre-processing: python data_pre_process.py
    2) To train and test model: python Neural_Net.py
 
