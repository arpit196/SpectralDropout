# example of loading the mnist dataset
from numpy import mean
from numpy import std
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.datasets import cifar10,cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
#from keras.layers.core import Lambda
#from keras import backend as K
from keras import regularizers
from keras.layers import LeakyReLU

import keras
import numpy as np


def load_images(cfr100=False):
    if(cfr100):
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images,noise=0.2)
    l=10
    if(cfr100):
        l=100
    train_labels = to_categorical(train_labels, l)
    test_labels = to_categorical(test_labels, l)

    return train_images, train_labels, test_images, test_labels

def normalization(train_images, test_images, noise=0.2):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    
    test_images = (test_images - mean) / (std + 1e-7)
    
    return train_images, test_images


lr_drop=20
learning_rate=0.1
def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))