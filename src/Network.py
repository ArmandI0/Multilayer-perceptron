import numpy as np
from .Layer import HiddenLayer, OutputLayer

class Network:
    def __init__(self, config: dict, nbOfFeatures: int):
        self.layer = []
        i = 0
        for param in config:
            initFunction = config[param]['weightInit']
            if param == '0':
                nbOfNeurons = int(config[param]['nbOfNeurons'])
                newLayer = HiddenLayer(nbOfNeurons, nbOfFeatures, initFunction)
            elif param == 'OutputLayer':
                newLayer = OutputLayer(2, self.layer[i - 1].size(), initFunction)
            else:
                nbOfNeurons = int(config[param]['nbOfNeurons'])
                newLayer = HiddenLayer(nbOfNeurons, self.layer[int(param) - 1].size(), initFunction)
            i += 1
            self.layer.append(newLayer)
        self.nbOfLayer = i

    def get_layer(self, layerNum: int):
        try:
            return self.layer[layerNum]
        except IndexError as e:
            raise ValueError('Invalid index: ', e)
    
    def doEpoch(self, batch, expectedOutput):
        i = 0
        for layer in self.layer:    
            if i == 0:
                A = layer.forwardPropagation(batch)
            else:
                A = layer.forwardPropagation(A)
            i += 1
        Error = layer.meanSquaredError(expectedOutput, A[:, 0])
        print(f"Layer {i}\n RETURN = \n{repr(A)}")
        print(f"ERROR = {repr(Error)}")

