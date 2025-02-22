import numpy as np
import math

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
        print("=" * 50)
        print(f"🟢 Hidden layer info : numberOfNeurons = {self.numberOfNeurons} / inputNb = {self.inputSize}")
        print("=" * 50)


        print(f"🔹 Shape of Weights: {self.weights.shape}")
        print(f"Weights:\n{repr(self.weights)}\n")

        print(f"🔹 Shape of Input A: {A.shape}")
        print(f"Input A:\n{repr(A)}\n")

        Z = np.dot(A, self.weights) + self.biais
        print("=" * 50)
        print(f"Z = sum(xi * wi + biais")
        print("=" * 50)
        print(f"🔹 Computed Z Matrix: {Z.shape} Z1 to Zn pour chaque neurone")
        print(f"Z Matrix:\n{repr(Z)}\n")
        print("=" * 50)

        return self.ReLU(Z)

    # Utils
    def size(self):
        return self.numberOfNeurons
    
    def getBiais(self):
        return self.bias
    
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
            return self.Z * (1 - self.Z)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        # Calcul de exp() pour chaque élément
        exp_Z = np.exp(Z)
        # Division par la somme pour normaliser
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forwardPropagation(self, A):
        print("=" * 50)
        print(f"🟢 Output layer info : numberOfNeurons = {self.numberOfNeurons} / inputNb = {self.inputSize}")
        print("=" * 50)


        print(f"🔹 Shape of Weights: {self.weights.shape}")
        print(f"Weights:\n{repr(self.weights)}\n")

        print(f"🔹 Shape of Input A: {A.shape}")
        print(f"Input A:\n{repr(A)}\n")

        Z = np.dot(A, self.weights) + self.biais

        print("=" * 50)
        print(f"Z = sum(xi * wi) + biais")
        print("=" * 50)
        print(f"🔹 Computed Z Matrix: {Z.shape} Z1 to Zn pour chaque neurone")
        print(f"Z Matrix:\n{repr(Z)}\n")
        print("=" * 50)


        Y = self.softmax(Z, False)
        print(Y[:, 0])
        self.A = A
        self.Z = Z[:, 0] # comme c'est un sortie binaire on peut garder seulement une colone
        self.Y = Y[:, 0]
        print('Z =', Z)
        return Y
    
    # dE_dw = dE_dz * y  et dE_dz = dE_dy * dérivée de la fonction d'activation dE_dy = dérivée de l'erreur par rapport à la sortie
    def backPropagation(self, yR):
        dE_dy = self.binaryCrossEntropyError(yR, True)
        print(dE_dy.shape , self.softmax(self.Z, True).shape)
        dE_dz = dE_dy * self.softmax(self.Z, True)
        dE_dzBis = yR - self.Y

        print(f'dE_dz {dE_dz} self.A {self.A.shape}')
        dE_dw = np.dot(self.A.T, dE_dz.reshape(-1,1))
        print(f'shape weight : {self.weights.shape} shape result : {dE_dw.shape}')
        print(f'weight: {self.weights} \n dE_dw : {dE_dw}')
        self.weights = self.weights - 0.05 * dE_dw
        print('self.weight\n', self.weights)

    def meanSquaredError(self, yR: np.array):
        E = (yR - self.Y)**2
        return E
    
    # derive partiel par rapport a expectedY
    def binaryCrossEntropyError(self, yR: np.array, derivate: bool):
        if derivate == True:
            dE = -1/len(yR) * (yR/self.Y - (1-yR)/(1-self.Y))
            return dE
        else:
            E = -1 / len(yR) * np.sum(yR * np.log(self.Y) + (1 - yR) * np.log(1 - self.Y))
        return E


