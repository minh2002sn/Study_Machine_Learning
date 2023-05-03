import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
from util import * 
from dnn_np import test
import pdb


def bat_classification_tf():
    # Load data from file
    # Make sure that bat.dat is in data/
    train_x, train_y, test_x, test_y = get_bat_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)    

    test_y  = test_y.flatten().astype(np.int32)
    train_y = train_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]
 
    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.0005
    momentum_rate = 0.005
    batch_size = 200
    steps = 5000
   
    # print(train_x[0::3])
    # print(train_x.shape)
    # Specify that all features have real-value data
    feature_columns = [tf1.feature_column.numeric_column("x", shape=[train_x.shape[1]])]


    # Available activition functions
    # https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    # tf1.nn.relu
    # tf1.nn.elu
    # tf1.nn.sigmoid
    # tf1.nn.tanh
    activation = tf.nn.softmax
    
    # [TODO 1.7] Create a neural network and train it using estimator

    # Some available gradient descent optimization algorithms
    # https://www.tensorflow.org/api_guides/python/train#Optimizers
    # tf1.train.GradientDescentOptimizer
    # tf1.train.AdadeltaOptimizer
    # tf1.train.AdagradOptimizer
    # tf1.train.AdagradDAOptimizer
    # tf1.train.MomentumOptimizer
    # tf1.train.AdamOptimizer
    # tf1.train.FtrlOptimizer
    # tf1.train.ProximalGradientDescentOptimizer
    # tf1.train.ProximalAdagradOptimizer
    # tf1.train.RMSPropOptimizer
    # Create optimizer

    # optimizer = tf1.train.GradientDescentOptimizer(learning_rate=0.01)
    optimizer = tf1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum_rate)
    
    # build a deep neural network
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    classifier = tf1.estimator.DNNClassifier(
        hidden_units=hidden_layers,
        feature_columns=feature_columns,
        n_classes=num_class,
        # optimizer=optimizer,
        # activation_fn=activation
    )
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf1.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf1.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False
    )
    
    # Evaluate accuracy. 
    predict_input_fn = tf1.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        num_epochs=1,
        shuffle=False
    )

    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)

    y_hat = create_one_hot(y_hat, num_class)
    visualize_point(test_x, test_y, y_hat)


def mnist_classification_tf():
    # Load data from file
    # Make sure that fashion-mnist/*.gz is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)
    train_x, val_x, test_x = normalize(train_x, train_x, test_x)    

    train_y = train_y.flatten().astype(np.int32)
    val_y = val_y.flatten().astype(np.int32)
    test_y = test_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]
    # pdb.set_trace()

    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.0005
    batch_size = 500
    steps = 1000
   
    print(train_x.shape)
    # Specify that all features have real-value data
    feature_columns = [tf1.feature_column.numeric_column("x", shape=[train_x.shape[1]])]
    # feature_columns = [tf1.feature_column.numeric_column("x", shape=[28, 28])]


    # Choose activation function
    activation = tf.nn.softmax
    
    # Some available gradient descent optimization algorithms 
    # TODO: [YC1.7] Create optimizer
    # optimizer = tf1.train.ProximalGradientDescentOptimizer(
    #     learning_rate=learning_rate,
    #     l2_regularization_strength=0.000001
    # )
    optimizer = tf1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.005)


    # build a deep neural network
    classifier = tf1.estimator.DNNClassifier(
        hidden_units=hidden_layers,
        feature_columns=feature_columns,
        n_classes=num_class,
        # optimizer=optimizer,
        # activation_fn=activation
    )
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf1.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf1.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)
    
    # Evaluate accuracy. 
    predict_input_fn = tf1.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        num_epochs=1,
        shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)


if __name__ == '__main__':
    np.random.seed(2017) 

    plt.ion()
    bat_classification_tf()
    # mnist_classification_tf()

    # pdb.set_trace()
