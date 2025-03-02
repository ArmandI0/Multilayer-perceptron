import numpy as np
import src.tools as tl
import json
from src.Network import Network

def main():
    try:
        df = tl.load_csv('data/data_test.csv')
        dfParams = tl.load_csv('data/normalisation_params.csv')

        answers = np.array([0 if x == 'B' else 1 for x in df.iloc[:, 0]])


        datas = tl.normalizeDatasWithParams(df, dfParams)
        with open('generated_config.json', 'r') as f:
            network = Network(json.load(f), datas.shape[1], datas, answers)
        with open('network_config.json', 'r') as f:
            params = json.load(f)
        network.configNetworkForPredict(params)
        accuracy = network.predictFonction()
        print(accuracy)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()