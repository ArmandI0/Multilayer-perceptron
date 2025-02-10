import numpy as np

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
        print(f"🟢 Forward Propagation - Layer Info")
        print("=" * 50)

        print(f"🔹 Shape of Weights: {self.weights.shape}")
        print(f"Weights:\n{repr(self.weights)}\n")

        print(f"🔹 Shape of Input A: {A.shape}")
        print(f"Input A:\n{repr(A)}\n")

        Z = np.dot(A, self.weights) + self.biais

        print("=" * 50)
        print(f"🔹 Computed Z Matrix: {Z.shape}")
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
            d = np.diag(Z)
            return d - np.outer(Z, Z)
        
        Z = Z - np.max(Z, axis=1, keepdims=True)
        # Calcul de exp() pour chaque élément
        exp_Z = np.exp(Z)
        # Division par la somme pour normaliser
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forwardPropagation(self, A):
        Z = np.dot(A, self.weights) + self.biais
        Y = self.softmax(Z, False)
        print(Y[:, 0])
        return Y
    
    def meanSquaredError(self, expectedYi, predictYi):
        print('expectedYi shape', expectedYi.shape, type(expectedYi), 'predictYi shape', predictYi.shape, type(predictYi))
        E = (expectedYi - predictYi)**2
        return E
    


