import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# reads image 'opencv-logo.png' as grayscale
path = 'E:\GitHub\machine_learning_examples\Test\Captcha'
for image in os.listdir(path):
    img = cv2.imread('{path}\\{image}'.format(path=path, image=image), 0)
    plt.imshow(img, cmap='gray')
    plt.show()