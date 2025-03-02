import pandas as pd
import math as mt

def load_csv(path: str) -> pd.DataFrame:
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

def splitDatasetRandomly(datas: pd.DataFrame):
    trainingSet = pd.DataFrame()
    testingSet = pd.DataFrame()
    randomData = datas.sample(frac=1)

    size = len(datas)
    mid = int(size / 2)

    trainingSet =  randomData.iloc[0:mid, :]
    testingSet = randomData.iloc[mid:size, :]
    trainingSet.to_csv('data/data_training.csv')
    testingSet.to_csv('data/data_test.csv')

def normalizePdSeries(variable : pd.Series, parameters : pd.Series) -> pd.Series :
    """
    Function to standardize a given variable from its different values
    Parameters : a pd.Series object containing the mean and std of the variable
    Return : a new pd.Series containing the normalized values of the variable
    """ 
    variableNormalized = (variable - parameters['mean']) / parameters['std']
    return variableNormalized

def normalizeDatas(datas: pd.DataFrame):
    entry = datas.iloc[:, 1:]
    entry = entry.astype(float)
    
    normalizeDatas = pd.DataFrame()
    normalisationParam = pd.DataFrame(
        index=['mean', 'std', 'median'],
        columns=range(len(entry.columns))
    )
    
    for i, serie in enumerate(entry.columns):
        mean = entry[serie].mean()
        std = entry[serie].std()
        median = entry[serie].median()
        
        normalisationParam[i] = [mean, std, median]
        normalizeDatas[i] = normalizePdSeries(entry[serie].fillna(median), normalisationParam[i])
    
    normalisationParam.to_csv('data/normalisation_params.csv', index=True, index_label='parameter')
    
    return normalizeDatas

def normalizeDatasWithParams(datas: pd.DataFrame, params: pd.DataFrame):
    entry = datas.iloc[:, 1:]
    entry = entry.astype(float)
    normalizeDatas = pd.DataFrame()
    for i, serie in enumerate(entry.columns):
        mean = params.loc['mean', str(i)]
        std = params.loc['std', str(i)]
        median = params.loc['std', str(i)]
        normalisationParam = pd.Series({
            'mean': mean,
            'std': std,
            'median': median
        })
        normalizeDatas[i] = normalizePdSeries(entry[serie].fillna(median), normalisationParam)
    return normalizeDatas

