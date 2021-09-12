# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def initRandomFilter(filter_size, featureMapIn, featureMapOut):
    # Khởi tạo random filter với dimension of filter và size của featureMapIn và Out với variance căn 2 / (.....)
    return (np.random.randn(filter_size, filter_size, featureMapIn, featureMapOut) * np.sqrt(
        2.0 / (filter_size * filter_size * featureMapIn))).astype(np.float32)


class ConvLayer:
    def __init__(self, filter_size, featureMapIn, featureMapOut, stride=2, padding='VALID'):
        # Khởi tạo ConvBlock
        # Khởi tạo giá trị filter và bias cho Mạng
        self.W = tf.Variable(initRandomFilter(filter_size, featureMapIn, featureMapOut))
        self.b = tf.Variable(np.zeros(featureMapOut, dtype=np.float32))
        self.stride = stride
        self.padding = padding

    # Trả về con_layer để tiến hành forward tới layer mới
    @tf.function
    def forward(self, X):
        X = tf.nn.conv2d(input=X, filters=self.W, strides=self.stride, padding=self.padding)
        return X

    def copyFromKerasLayers(self, layer):
        # only 1 layer to copy from
        W, b = layer.get_weights()
        self.W.assign(W)
        self.b.assign(b)

    # Get params Function để khởi tạo params cho model
    def get_params(self):
        return [self.W, self.b]


class BatchNormLayer:
    def __init__(self, FeatureMapSize):
        self.running_mean = tf.Variable(np.zeros(FeatureMapSize, dtype=np.float32), trainable=False)
        self.running_var = tf.Variable(np.ones(FeatureMapSize, dtype=np.float32), trainable=False)
        self.gamma = tf.Variable(np.ones(FeatureMapSize, dtype=np.float32))
        self.beta = tf.Variable(np.zeros(FeatureMapSize, dtype=np.float32))

    @tf.function
    def forward(self, X):
        X = tf.nn.batch_normalization(X, self.running_mean, self.running_var, self.gamma, self.beta,
                                      variance_epsilon=1e-3)
        return X

    def copyFromKerasLayers(self, layer):
        # only 1 layer to copy from
        # order:
        # gamma, beta, moving mean, moving variance
        gamma, beta, running_mean, running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)

    def get_params(self):
        return [self.running_mean, self.running_var, self.gamma, self.beta]


class ConvBlock:
    def __init__(self, inputDimension, featureMapSizes, stride=2, activation_function=tf.nn.relu):
        # Conv Layer 1 , 2 ,3
        # Feature map shortcut -> output = feature map size 3

        # note: kernel size in 2nd conv is always 3
        #       so we won't bother including it as an arg

        # note: stride only applies to conv 1 in main branch
        #       and conv in shortcut, otherwise stride is 1

        self.activation_fun = activation_function

        # Khởi tạo nhánh chính
        # Conv -> BN -> F() -> Conv -> BN -> F() -> Conv -> BN
        self.conv1 = ConvLayer(filter_size=1, featureMapIn=inputDimension, featureMapOut=featureMapSizes[0],
                               stride=stride)
        self.bn1 = BatchNormLayer(FeatureMapSize=featureMapSizes[0])
        self.conv2 = ConvLayer(filter_size=3, featureMapIn=featureMapSizes[0], featureMapOut=featureMapSizes[1],
                               stride=1, padding='SAME')
        self.bn2 = BatchNormLayer(FeatureMapSize=featureMapSizes[1])
        self.conv3 = ConvLayer(filter_size=1, featureMapIn=featureMapSizes[1], featureMapOut=featureMapSizes[2],
                               stride=1)
        self.bn3 = BatchNormLayer(FeatureMapSize=featureMapSizes[2])

        # Khởi tạo nhánh shortcut
        # Conv -> BN để chuyển thành dạng cùng với output
        self.convs = ConvLayer(filter_size=1, featureMapIn=inputDimension, featureMapOut=featureMapSizes[2],
                               stride=stride)
        self.bns = BatchNormLayer(FeatureMapSize=featureMapSizes[2])

        # Tổng hợp lại thành layers
        self.layers = [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.conv3, self.bn3,
            self.convs, self.bns
        ]

    @tf.function
    def forward(self, X):
        # Nhánh chính
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)

        # Nhánh shortcut
        SX = self.convs.forward(X)
        SX = self.bns.forward(SX)

        # Cộng hai nhánh lại
        return self.activation_fun(FX + SX)

    @tf.function
    def predict(self, X):
        return self.forward(X)


    def copyFromKerasLayers(self, layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[8])
        self.convs.copyFromKerasLayers(layers[7])
        self.bns.copyFromKerasLayers(layers[9])

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


if __name__ == '__main__':
    conv_block = ConvBlock(inputDimension=3, featureMapSizes=[64, 64, 256], stride=1)

    # make a fake image
    X = np.random.random((1, 224, 224, 3))
    output = conv_block.predict(X)
    print("output.shape:", output.shape)
