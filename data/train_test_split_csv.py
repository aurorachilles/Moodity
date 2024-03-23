import pandas as pd
import numpy as np
from tqdm import tqdm
def emotion_spliter_train(
    data: pd.DataFrame,
    train_size: float,
    count: int,
) -> pd.DataFrame:
    train_num = round(count*train_size)
    # print(train_num)
    train_main = pd.concat([train_main,data[:train_num]])
    return train_main

def emotion_spliter_test(
    data: pd.DataFrame,
    test_size: float,
    count: int,
) -> pd.DataFrame:
    train_num = round(count*test_size)
    # print(train_num)
    test_main = data[train_num:]
    return test_main

def main():
    data = pd.read_csv('./dataset/labels.csv').drop(columns=['Unnamed: 0'])
    # print(data.shape)
    emotions = pd.read_csv('./dataset/emotion_count.csv').drop(columns=['Unnamed: 0'])
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_size = float(input("Enter a train_test split size:"))
    print("Splitting Data")
    for i in tqdm(range(emotions.shape[0])):
        emotion = emotions.iloc[i,0]
        emotion_data = data.loc[data['label'] == emotion]
        num_for_split = round(emotion_data.shape[0]* train_size)

        train_data = pd.concat([train_data, emotion_data[:num_for_split]] ,ignore_index=True)
        test_data = pd.concat([test_data, emotion_data[num_for_split:]], ignore_index=True)

    print("Saving To CSV")
    train_data.to_csv('./dataset/train_data_labels.csv')
    test_data.to_csv('./dataset/test_data_labels.csv')
    print("Finished Savings to CSV")


if __name__ == "__main__":
    main()