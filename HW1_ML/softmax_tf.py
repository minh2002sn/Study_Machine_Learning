"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *
import time

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    np.random.seed(2020)
    tf.set_random_seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # print(train_x.shape)
    # print(train_y.shape)

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    x = tf.placeholder(dtype=tf.float32, shape=[None, train_x.shape[1]])
    y = tf.placeholder(dtype=tf.float32, shape=[None, train_y.shape[1]])

    # [TODO 2.8] Create weights (W) using TF variables 
    W = tf.Variable(tf.zeros([train_x.shape[1], train_y.shape[1]]))

    # [TODO 2.9] Create a feed-forward operator 
    logits = tf.matmul(x, W)
    pred = tf.nn.softmax(logits)

    # [TODO 2.10] Write the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

    # Define hyper-parameters and train-related parameters
    num_epoch = 1000
    # num_epoch = 3347
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.perf_counter()
            # [TODO 2.8] Compute losses and update weights here
            train_loss = sess.run(cost, feed_dict={x:train_x, y:train_y}) 
            val_loss = sess.run(cost, feed_dict={x:val_x, y:val_y}) 
            # Update weights
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            toc = time.perf_counter()
            # print(toc-tic)
            # [TODO 2.11] Define your own stopping condition here 
          
        
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
