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
        result = np.array(df.iloc[0:3, 0])
        result = np.array([0 if x == 'B' else 1 for x in result])
        test = dataN.iloc[0:3, :]
        
        with open('generated_config.json', 'r') as data_file:
            networkConfig = json.load(data_file)
        network = Network(networkConfig, 5)
        batch = np.array(test) # 3, 30

        # print(batch)
        for i in range(50):
            network.doEpoch(batch, result)

    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()