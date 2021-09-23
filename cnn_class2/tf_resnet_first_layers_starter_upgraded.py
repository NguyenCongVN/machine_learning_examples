# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions

from tf_resnet_convblock_starter import ConvLayer, BatchNormLayer, ConvBlock
tf.compat.v1.disable_v2_behavior()

class ReLULayer:
    @tf.function
    def forward(self, X):
        return tf.nn.relu(X)


class MaxPoolLayer:
    def __init__(self, windowSize):
        self.windowSize = windowSize

    @tf.function
    def forward(self, X):
        return tf.nn.max_pool2d(input=X, ksize=[1, self.windowSize, self.windowSize, 1], strides=[1, 2, 2, 1],
                                padding='VALID')


class ZeroPaddingLayer:
    def __init__(self, paddingSize):
        self.paddingSize = paddingSize
    @tf.function
    def forward(self, X):
        return tf.keras.layers.ZeroPadding2D(padding=self.paddingSize)(X)


class PartialResNet:
    def __init__(self):
        # Khởi tạo các layers
        self.layers = [
            ConvLayer(filter_size=7, featureMapIn=3, featureMapOut=64, stride=2, padding='SAME'),
            BatchNormLayer(FeatureMapSize=64),
            ReLULayer(),
            ZeroPaddingLayer(paddingSize=1),
            MaxPoolLayer(windowSize=3),
            # Conv Block
            ConvBlock(inputDimension=64, featureMapSizes=[64, 64, 256], stride=1)
        ]

    @tf.function
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def copyFromKerasLayers(self, layers):
        self.layers[0].copyFromKerasLayers(layers[2])
        self.layers[1].copyFromKerasLayers(layers[3])
        self.layers[5].copyFromKerasLayers(layers[7:])

    @tf.function
    def predict(self, X):
        # Lưu X vào
        return self.forward(X)


if __name__ == '__main__':
    # you can also set weights to None, it doesn't matter
    resnet = ResNet50(weights='imagenet')

    # you can determine the correct layer
    # by looking at resnet.layers in the console
    partial_model = Model(
        inputs=resnet.input,
        outputs=resnet.layers[18].output
    )
    print(partial_model.summary())
    # for layer in partial_model.layers:
    #   layer.trainable = False

    # Model đã tạo để so sánh với model thực trong thư viện
    my_partial_resnet = PartialResNet()
    #
    # make a fake image
    X = np.random.random((1, 224, 224, 3)).astype('float32')
    # get keras output
    keras_output = partial_model.predict(X)

    # note: starting a new session messes up the Keras model
    # session = keras.backend.get_session()
    # my_partial_resnet.set_session(session)
    # session.run(init)

    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print("first_output.shape:", first_output.shape)

    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)

    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = tf.math.reduce_sum(tf.math.abs(output - keras_output)).numpy()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)
