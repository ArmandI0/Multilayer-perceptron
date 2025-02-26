import numpy as np
import src.tools as tl
import json
from src.Network import Network

def main():
    try:
        # Chargement et préparation des données
        df = tl.load_csv('data/testing_set.csv', header=0)
        X = tl.normalize_datas(df).values
        y = np.array([0 if x == 'B' else 1 for x in df.iloc[:, 0]])
        
        # Division en batches de taille 32
        batchSize = 5
        nbBatchs = len(X) // batchSize
        print(nbBatchs)
        # Création des batches
        A = np.array_split(X, nbBatchs)
        yR = np.array_split(y, nbBatchs)
        
        # Création du réseau
        with open('generated_config.json', 'r') as f:
            network = Network(json.load(f), X.shape[1])
        
        # Entraînement
        for epoch in range(1):
            # Entraînement sur chaque batch
            for i in range(nbBatchs):
                network.doEpoch(A[i], yR[i])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()