import numpy as np
import math
from .Logger import Logger

class NeuralLayer:
    def __init__(self, sizeOfLayer: int, inputSize: int, initFunction: str):
        self.inputSize = inputSize
        self.numberOfNeurons = sizeOfLayer
        self.biais = np.zeros(sizeOfLayer).reshape(-1,1).T
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

class HiddenLayer(NeuralLayer):

    def __init__(self, sizeOfLayer: int, inputSize: int, initFunction: str):
        super().__init__(sizeOfLayer, inputSize, initFunction)


    # Forward propagation

    def forwardPropagation(self, A):

        Z = np.dot(A, self.weights) + self.biais
        Y = self.ReLU(Z)

        log = Logger.getInstance()
        log.logForward('Hidden layer', A, self.numberOfNeurons, self.weights, Z, Y)
        return Y
    
    def backPropagation(self, dE_dZoutput):
        dE_dZ = np.dot(dE_dZoutput, self.weights.T)
        print('test', dE_dZ)

    # Utils
    def size(self):
        return self.numberOfNeurons
    
    def getBiais(self):
        return self.biais
    
    def getWeights(self):
        return self.weights

    # Fonctons d'activations

    def ReLU(self, X):
        return np.maximum(0, X)
    
class OutputLayer(NeuralLayer):
    def __init__(self, outputSize: int, inputSize: int, initFunction: str):
        super().__init__(outputSize, inputSize, initFunction)

    def softmax(self, Z, derivate: bool):
        if derivate == True:
            return self.Y * (1 - self.Y)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        # Calcul de exp() pour chaque élément
        exp_Z = np.exp(Z)
        # Division par la somme pour normaliser
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forwardPropagation(self, A):
        Z = np.dot(A, self.weights) + self.biais
        Y = self.softmax(Z, False)

        self.A = A
        self.Z = Z[:, 0] # comme c'est un sortie binaire on peut garder seulement une colone
        self.Y = Y[:, 0]

        log = Logger.getInstance()
        log.logForward('Output layer', A, self.numberOfNeurons, self.weights, Z, Y)
    
        return Y
    
    # dE_dw = dE_dz * y  et dE_dz = dE_dy * dérivée de la fonction d'activation dE_dy = dérivée de l'erreur par rapport à la sortie
    def backPropagation(self, yR):
        dE_dy = self.binaryCrossEntropyError(yR, True)
        dE_dz = dE_dy * self.softmax(self.Z, True)
        dE_dzBis = yR - self.Y
        print(f"🔹dE_dz: \n{repr(dE_dz)}\n 🔹dE_dzBis: \n {repr(dE_dzBis)}")

        dE_dw = np.dot(self.A.T, dE_dz.reshape(-1,1))
        oldWeight = self.weights.copy()
        self.weights = self.weights - 0.05 * dE_dw
        print('self.weight\n', self.weights)
        log = Logger.getInstance()
        log.logBackward('Output layer', dE_dz, dE_dw, oldWeight, self.weights)
        
        return dE_dz

    def meanSquaredError(self, yR: np.array):
        E = (yR - self.Y)**2
        return E
    
    # derive partiel par rapport a expectedY
    def binaryCrossEntropyError(self, yR: np.array, derivate: bool):
        if derivate:
            return -(yR / self.Y) + ((1 - yR) / (1 - self.Y))
        else:
            E = -1 / len(yR) * np.sum(yR * np.log(self.Y) + (1 - yR) * np.log(1 - self.Y))
        return E


