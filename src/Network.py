import numpy as np
from .Layer import Layer

class Network:
    def __init__(self, config: dict, nbOfFeatures: int):
        self.layer = []
        for param in config:
            nbOfNeurons = int(config[param]['nbOfNeurons'])
            initFunction = config[param]['weightInit']
            if param == '0':
                newHiddenLayer = Layer(nbOfNeurons, nbOfFeatures, initFunction)
            else:
                newHiddenLayer = Layer(nbOfNeurons, self.layer[int(param) - 1].size(), initFunction)
            self.layer.append(newHiddenLayer)

    def get_layer(self, layerNum: int):
        try:
            return self.layer[layerNum]
        except IndexError as e:
            raise ValueError('Invalid index: ', e)
    
    def doEpoch(self, batch):
        for neuron in self.layer:
            neuron.forwardPropagation(batch)