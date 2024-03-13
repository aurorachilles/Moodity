import pandas as pd
from tqdm import tqdm


def main():
    print("Reading Data")
    data = pd.read_csv('./dataset/new_labels.csv')
    cols = list(data['label'].unique())
    count = [0 for _ in cols]

    print("starting count")
    for row in tqdm(range(len(data))):
        data_col = data.iloc[row, 2]
        count[cols.index(data_col)] += 1

    print("saving data")
    dataset = pd.DataFrame({'emotion_name': cols, 'count': count})
    dataset.to_csv('./dataset/emotion_count.csv')
    print("data saved!")


if __name__ == "__main__":
    main()
