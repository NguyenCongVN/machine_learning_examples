from __future__ import print_function, division
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self , eps=0.1 , alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.state_history = []
    def update_history(self , s):

