import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import *
from keras.models import Sequential, Model
import time


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

simple_DNN = keras.Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(units=100, activation='relu'),
    layers.Dense(units=10, activation='relu'),
    layers.Dense(num_classes, activation="softmax")
])

simple_DNN.summary()

batch_size = 128
epochs = 15


simple_DNN.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
simple_DNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


medium_DNN = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(units=num_classes, activation='softmax')
])

medium_DNN.summary()

medium_DNN.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
medium_DNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

time.sleep(15)

(xtrain,ytrain),(xtest,ytest)= keras.datasets.mnist.load_data()
xtrain = x_train.astype("float32") / 255

xtrain=np.dstack([xtrain] * 3)
xtest=np.dstack([xtest]*3)

xtrain = xtrain.reshape(-1, 28,28,3)
xtest= xtest.reshape (-1,28,28,3)

from keras.preprocessing.image import img_to_array, array_to_img

xtrain = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in xtrain])
xtest = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in xtest])

vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
vgg16_altered = Sequential()
for layer in vgg16_model.layers[:-1]:
    vgg16_altered.add(layer)

for layer in vgg16_altered.layers:
    layer.trainable = False

flatten = Flatten()
new_layer2 = Dense(10, activation='softmax', name='my_dense_2')

inp2 = vgg16_altered.input
out2 = new_layer2(flatten(vgg16_altered.output))
vgg16_altered = Model(inp2, out2)

vgg16_altered.summary()

vgg16_altered.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
vgg16_altered.fit(xtrain, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


print("Saving the models")
simple_DNN.save('models/simple_DNN_for_mnist.h5')
medium_DNN.save('models/medium_DNN_for_mnist.h5')
vgg16_altered.save('models/large_DNN_for_mnist.h5')
