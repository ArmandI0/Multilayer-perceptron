import numpy as np


class Layer:

    def __init__(self, sizeOfLayer: int, inputSize: int, initFunction: str):
        self.inputSize = inputSize
        self.numberOfNeurons = sizeOfLayer
        self.biais = np.zeros(inputSize)
        self.weights = self.initFunctions[initFunction](self) 

    # Fonctions d'initialisations des poids
    def uniform_init(self):
        return np.random.uniform(-0.05, 0.05, size=(self.inputSize, self.numberOfNeurons))

    def glorot_uniform(self):
        limit = np.sqrt(6/(self.inputSize + self.numberOfNeurons))
        return np.random.uniform(-limit, limit, size=(self.inputSize, self.numberOfNeurons))
    
    def he_normal(self):
        return np.random.normal(0, np.sqrt(2/self.inputSize), size=(self.inputSize, self.numberOfNeurons))
    
    def lecun_uniform(self):
        limit = np.sqrt(3/self.inputSize)
        return np.random.uniform(-limit, limit, size=(self.inputSize, self.numberOfNeurons))
    
    initFunctions = {
        'uniform_init': lambda self: self.uniform_init(),
        'glorot_uniform': lambda self: self.glorot_uniform(),
        'he_normal': lambda self: self.he_normal(),
        'lecun_uniform': lambda self: self.lecun_uniform(),
    }

    # Forward propagation

    def forwardPropagation(self, A):
        print('shape weight = ', self.weights.shape, 'shape A = ', A.shape, 'shape biais', self.biais.shape)
        Z = np.dot(self.weights, A) + self.biais
        self.ReLU(Z)

    # Utils
    def size(self):
        return self.numberOfNeurons
    
    def getBiais(self):
        return self.bias
    
    def getWeights(self):
        return self.weights

    # Fonctons d'activations

    def ReLU(self, X):
        print(max(0, X))
        return max(0, X)