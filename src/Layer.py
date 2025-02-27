import numpy as np
import math
from .Logger import Logger

class NeuralLayer:
    def __init__(self, sizeOfLayer: int, inputSize: int, initFunction: str):
        self.inputSize = inputSize
        self.numberOfNeurons = sizeOfLayer
        self.biais = np.zeros(sizeOfLayer).reshape(-1,1).T
        self.weights = self.initFunctions[initFunction](self)
        self.learningRate = 0.01

    # Utils
    def size(self):
        return self.numberOfNeurons
    
    def getBiais(self):
        return self.biais
    
    def getWeights(self):
        return self.weights

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

    # Fonctons d'activations

    def ReLU(self, X, derivate: bool):
        if derivate:
            return (X > 0).astype(float)
        else:
            return np.maximum(0, X)

    # Forward propagation

    def forwardPropagation(self, A):
        self.A = A
        Z = np.dot(A, self.weights) + self.biais
        self.Z = Z
        Y = self.ReLU(Z, False)
        self.Y = Y

        log = Logger.getInstance()
        log.logForward('Hidden layer', A, self.numberOfNeurons, self.weights, Z, Y)

        return Y
    
    def backPropagation(self, dE_dz):
        deriveActivationLayer = self.ReLU(self.Z, True)
        delta = dE_dz * deriveActivationLayer
        gradient = np.dot(self.A.T, delta)
        self.weights = self.weights - self.learningRate * gradient
        self.biais = self.biais - self.learningRate * np.sum(dE_dz, axis=0)
        dE_dzLayer = np.dot(delta, self.weights.T)
        return dE_dzLayer

    
class OutputLayer(NeuralLayer):
    def __init__(self, outputSize: int, inputSize: int, initFunction: str):
        super().__init__(outputSize, inputSize, initFunction)
        self.learning_rate = 0.001  # Réduit

    def softmax(self, Z, derivate: bool):
        if derivate:
            return self.Y * (1 - self.Y)
        
        # Normalisation pour éviter l'explosion numérique
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / (np.sum(exp_Z, axis=1, keepdims=True) + 1e-15)
   
    def forwardPropagation(self, A):
        Z = np.dot(A, self.weights) + self.biais
        Y = self.softmax(Z, False)

        self.A = A
        self.Z = Z
        self.Y = Y

        log = Logger.getInstance()
        log.logForward('Output layer', A, self.numberOfNeurons, self.weights, Z, Y)
        # np.set_printoptions(suppress=True)
        # print(f"output layer {Y}")
        return Y
    
    # dE_dw = dE_dz * y  et dE_dz = dE_dy * dérivée de la fonction d'activation dE_dy = dérivée de l'erreur par rapport à la sortie
    def backPropagation(self, yR):
        # Convertir les étiquettes en format one-hot
        yRReshape = np.eye(2)[yR]
        
        # Pour softmax + cross-entropy, le gradient est (y_pred - y_true)
        dE_dz = self.Y - yRReshape  # Notez le changement d'ordre!
        
        # Calcul du gradient pour les poids
        dE_dw = np.dot(self.A.T, dE_dz)
        
        # Mise à jour des poids et biais avec un learning rate faible
        self.weights -= self.learning_rate * dE_dw
        self.biais -= self.learning_rate * np.sum(dE_dz, axis=0)
        
        # Propagation de l'erreur à la couche précédente
        return np.dot(dE_dz, self.weights.T)

    def meanSquaredError(self, yR: np.array):
        E = (yR - self.Y)**2
        return E
    
    # Fonction de cout / loss function
    def binaryCrossEntropyError(self, yR: np.array, derivate: bool):
        epsilon = 1e-15  # Évite la division par zéro
        Y_safe = np.clip(self.Y, epsilon, 1 - epsilon)
        if derivate:
            return -(yR / Y_safe) + ((1 - yR) / (1 - Y_safe))
        else:
            # Conversion en one-hot si nécessaire
            if len(yR.shape) == 1:
                yR_one_hot = np.eye(2)[yR]
            else:
                yR_one_hot = yR
                
            E = -1 / len(yR) * np.sum(yR_one_hot * np.log(Y_safe) + (1 - yR_one_hot) * np.log(1 - Y_safe))
        return E


