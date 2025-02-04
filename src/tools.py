import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Function to load a csv
    Parameters : path of the csv
    Return : pd.DataFrame containing the csv datass
    """
    try:
        csv = pd.read_csv(path)
    except Exception as e:
        print(f"loading csv error : {e}")
        return None
    return csv

def split_dataset_randomly(datas: pd.DataFrame):
    trainingSet = pd.DataFrame()
    testingSet = pd.DataFrame()
    randomData = datas.sample(frac=1)

    print(randomData)
    trainingSet =  randomData.iloc[:, 2]
    print(trainingSet)

df = load_csv('data/data.csv')
split_dataset_randomly(df)
    
