import pandas as pd
import math as mt

def load_csv(path: str, header: int) -> pd.DataFrame:
    """
    Function to load a csv
    Parameters : path of the csv
    Return : pd.DataFrame containing the csv datass
    """
    try:
        csv = pd.read_csv(path, header=0, index_col=0)
    except Exception as e:
        print(f"loading csv error : {e}")
        return None
    return csv

def split_dataset_randomly(datas: pd.DataFrame):
    trainingSet = pd.DataFrame()
    testingSet = pd.DataFrame()
    randomData = datas.sample(frac=1)

    size = len(datas)
    mid = int(size / 2)

    trainingSet =  randomData.iloc[0:mid, :]
    testingSet = randomData.iloc[mid:size, :]
    trainingSet.to_csv('data/training_set.csv')
    testingSet.to_csv('data/testing_set.csv')

def normalizePdSeries(variable : pd.Series, parameters : pd.Series) -> pd.Series :
    """
    Function to standardize a given variable from its different values
    Parameters : a pd.Series object containing the mean and std of the variable
    Return : a new pd.Series containing the normalized values of the variable
    """ 
    variableNormalized = (variable - parameters['mean']) / parameters['std']
    return variableNormalized

def normalize_datas(datas: pd.DataFrame):
    entry = datas.drop(columns='1')
    normalizeDatas = pd.DataFrame()
    normalisationParam = pd.DataFrame(index=['mean', 'std', 'median'])
    for serie in entry.columns:
        median = entry[serie].median()
        mean = entry[serie].mean()        
        std = entry[serie].std()
        normalisationParam[serie] = [median, mean, std]
        normalizeDatas[serie] = normalizePdSeries(entry[serie].fillna(median), normalisationParam[serie])
    normalisationParam.to_csv('data/normalistaion_params.csv')
    return normalizeDatas

        


# df = load_csv('data/data.csv')
# print(df.describe())
# normalize_datas(df)
    
