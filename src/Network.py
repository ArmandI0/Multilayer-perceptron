import numpy as np
import json
from src.Layer import HiddenLayer, OutputLayer
import matplotlib.pyplot as plt


class Network:
    def __init__(self, config: dict, nbOfFeatures: int, dataTest=None, answersTest=None):
        self.layer = []
        self.loss = []
        self.lossDataTest = []
        self.accuracy = []
        self.accuracyDataTest = []
        self.accuracy = []
        self.dataTest = dataTest
        self.answersTest = answersTest
        self.fig, self.axs = plt.subplots(2, 1)
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
        A = batch
        for layer in self.layer:
            A = layer.forwardPropagation(A, False)
        
        accuracy = self.layer[-1].fctAccuracy(expectedOutput)
        dE_dZ = self.layer[-1].backPropagation(expectedOutput)
        loss = self.layer[-1].binaryCrossEntropyError(expectedOutput, derivate=False)
        for layer in reversed(self.layer[:-1]):
            dE_dZ = layer.backPropagation(dE_dZ)
        return loss, accuracy

    def networkTraining(self, data, answers, batchSize, nbOfEpoch: int):
        nbBatchs = len(data) // batchSize
        
        # Création des batch
        A = np.array_split(data, nbBatchs)
        yR = np.array_split(answers, nbBatchs)
        for epoch in range(nbOfEpoch):
            batchLoss = 0
            batchAccuracy = 0
            # Entraînement sur chaque batch 
            for i in range(nbBatchs):
                a, b = self.doEpoch(A[i], yR[i])
                batchLoss += a
                batchAccuracy += b
            loss = batchLoss / nbBatchs
            accuracy = batchAccuracy / nbBatchs
            self.predictFonction()
            self.loss.append(loss)
            self.accuracy.append(accuracy)
            print(f"epoch : {epoch + 1}/{nbOfEpoch} - loss: {loss} - accuracy: {accuracy}")
    
    def lossCurveCreate(self):
        self.axs[0].plot(range(len(self.loss)), self.loss, 'b-', label='Training Loss')
        self.axs[0].plot(range(len(self.lossDataTest)), self.lossDataTest, 'r--', label='lossDatatest')
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('Loss')
        self.axs[0].grid(True)
        self.axs[0].legend()

    def accuracyCurveCreate(self):
        self.axs[1].plot(range(len(self.accuracy)), self.accuracy, 'b-', label='Training accuracy')
        self.axs[1].plot(range(len(self.accuracyDataTest)), self.accuracyDataTest, 'r--', label='dataTest')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Accuracy')
        self.axs[1].grid(True)
        self.axs[1].legend()

    def predictFonction(self):
        A = self.dataTest
        for layer in self.layer:
            A = layer.forwardPropagation(A, True)
        self.accuracyDataTest.append(self.layer[-1].fctAccuracyTest(self.answersTest, A))
        self.lossDataTest.append(self.layer[-1].binaryCrossEntropyErrorForPredict(self.answersTest, A))
        return self.accuracyDataTest[0]
    
    def saveNetworkAsJson(self):
        networkJson = {
            'weights': [],
            'biases': [],
        }
        for i, layer in enumerate(self.layer):
            networkJson['weights'].append(layer.getWeights().tolist())
            networkJson['biases'].append(layer.getBiais().tolist())
        with open('network_config.json', 'w') as f:
            json.dump(networkJson, f, indent=4)
    
    def configNetworkForPredict(self, params):
        for i, layer in enumerate(self.layer):
            # Conversion en numpy arrays
            weights = np.array(params['weights'][i])
            biases = np.array(params['biases'][i])
            layer.setWeights(weights)
            layer.setBiais(biases)
        
