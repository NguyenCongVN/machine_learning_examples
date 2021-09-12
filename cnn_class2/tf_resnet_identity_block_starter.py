# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np

from tf_resnet_convblock_starter import ConvLayer, BatchNormLayer


class IdentityBlock:
    def __init__(self, inputDimension, featureMapSize, activation_function=tf.nn.relu):
        self.activation_function = activation_function
        # Khởi tạo các layers
        # Nhánh chính
        # conv -> bn -> F -> conv -> bn -> F -> conv -> bn
        self.conv1 = ConvLayer(filter_size=1, featureMapIn=inputDimension, featureMapOut=featureMapSize[0],
                               stride=1)
        self.bn1 = BatchNormLayer(FeatureMapSize=featureMapSize[0])
        self.conv2 = ConvLayer(filter_size=3, featureMapIn=featureMapSize[0], featureMapOut=featureMapSize[1],
                               padding='SAME', stride=1)
        self.bn2 = BatchNormLayer(FeatureMapSize=featureMapSize[1])
        self.conv3 = ConvLayer(filter_size=1, featureMapIn=featureMapSize[1], featureMapOut=featureMapSize[2],
                               stride=1)
        self.bn3 = BatchNormLayer(FeatureMapSize=featureMapSize[2])

        # Lưu lại các layers
        self.layers = [self.conv1,
                       self.bn1,
                       self.conv2,
                       self.bn2,
                       self.conv3,
                       self.bn3]

    def forward(self, X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)

        # Cộng hai nhánh và trả lại kết quả
        return self.activation_function(FX + X)

    def predict(self, X):
        return self.forward(X)

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


if __name__ == '__main__':
    identity_block = IdentityBlock(inputDimension=256, featureMapSize=[64, 64, 256])

    # make a fake image
    X = np.random.random((1, 224, 224, 256))
    output = identity_block.predict(X)
    print("output.shape:", output.shape)
