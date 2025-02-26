import numpy as np
import src.tools as tl
import json
from src.Network import Network
import matplotlib.pyplot as plt

def main():
    try:
        # Chargement et préparation des données
        df = tl.load_csv('data/testing_set.csv', header=0)
        X = tl.normalize_datas(df).values
        y = np.array([0 if x == 'B' else 1 for x in df.iloc[:, 0]])
        
        # Division en batches de taille 32
        batchSize = 32
        nbBatchs = len(X) // batchSize
        
        # Création des batches
        A = np.array_split(X, nbBatchs)
        yR = np.array_split(y, nbBatchs)
        
        # Création du réseau
        with open('generated_config.json', 'r') as f:
            network = Network(json.load(f), X.shape[1])
        
        # Liste pour stocker les losses
        losses = []
        
        # Entraînement
        for epoch in range(100):
            epoch_loss = 0
            # Entraînement sur chaque batch 
            for i in range(nbBatchs):
                epoch_loss += network.doEpoch(A[i], yR[i])
            average_loss = epoch_loss / nbBatchs
            losses.append(average_loss)
            print(f"EPOCH = {epoch}, loss = {average_loss}")
        
        # Création du graphique
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Sauvegarde du graphique
        plt.savefig('training_loss.png')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()