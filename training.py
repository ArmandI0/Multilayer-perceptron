import numpy as np
import src.tools as tl
import os
import json
from src.Network import Network

def main():
    try :
        # if len(sys.argv) != 3:
        #     print('Error : dataset to predict and weight.json needed')
        #     return 1
        df = tl.load_csv('data/training_set.csv', header=0)
        dataN = tl.normalize_datas(df)
        test = dataN.iloc[0:10, :]
        with open('generated_config.json', 'r') as data_file:
            networkConfig = json.load(data_file)
        network = Network(networkConfig, 30)
        network.doEpoch(test.T)

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()