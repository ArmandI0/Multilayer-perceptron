import src.tools as tl


def main():
    try:
        df = tl.load_csv('data/data.csv')
        tl.splitDatasetRandomly(df)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()