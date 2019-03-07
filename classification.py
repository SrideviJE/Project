from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
​
import numpy as np
import matplotlib.pyplot as plt
​
print(tf.__version__)
​
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
​
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
​
train_images = train_images / 255.0
test_images = test_images / 255.0

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D,MaxPool2D
from keras.optimizers import RMSprop
class CnnIsFakeModel():
    # tuning parameter is defined within the __init__ for the sake of simplicity
    def __init__(self, inputshape):
        model = Sequential()

        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu', input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))


        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(512, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation = "softmax"))
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        self.model = model
    
    def getModel(self):
        return self.model
