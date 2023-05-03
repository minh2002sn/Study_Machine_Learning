import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pdb
from dnn_tf import *

if __name__ == '__main__':
    np.random.seed(2017)
    plt.ion()
    mnist_classification_tf()

    # pdb.set_trace()
