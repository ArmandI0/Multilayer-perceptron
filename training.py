import numpy as np
import tools as tl


def main():
    try :
        df = tl.load_csv('data/training_set.csv')
        df = tl.normalize_datas(df)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__== "__main__":
    main()