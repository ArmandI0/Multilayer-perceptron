import numpy as np
import src.tools as tl
import json
from src.Network import Network
import matplotlib.pyplot as plt

def main():
    try:
        # Chargement et préparation des données
        dfTest = tl.load_csv('data/testing_set.csv')
        dfData = tl.load_csv('data/training_set.csv')

        answersTraining = np.array([0 if x == 'B' else 1 for x in dfData.iloc[:, 0]])
        answersTesting = np.array([0 if x == 'B' else 1 for x in dfTest.iloc[:, 0]])

        trainingData = tl.normalizeDatas(dfData).values
        
        dfParams = tl.load_csv('data/normalisation_params.csv')
        testingData = tl.normalizeDatasWithParams(dfTest, dfParams).values

        # Network creation
        with open('generated_config.json', 'r') as f:
            network = Network(json.load(f), trainingData.shape[1], testingData, answersTesting)
        
        network.networkTraining(trainingData, answersTraining, 32, 20)
        network.accuracyCurveCreate()
        network.lossCurveCreate()
        network.saveNetworkAsJson()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()