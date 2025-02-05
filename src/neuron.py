import numpy as np


class Layer:
    def __init__(self, numberOfNeurons):
        self.weights = np.array()

    def random_normal_init(self):
        self.weights = self.weights.random.normal()
