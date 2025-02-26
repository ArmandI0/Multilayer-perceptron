import numpy as np
from src.Layer import HiddenLayer, OutputLayer


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
        # Forward pass
        A = batch
        for layer in self.layer:
            A = layer.forwardPropagation(A)
        # Backward pass
        dE_dZ = expectedOutput
        for layer in reversed(self.layer):
            dE_dZ = layer.backPropagation(dE_dZ)

    def networkTraining(self, data):
        pass