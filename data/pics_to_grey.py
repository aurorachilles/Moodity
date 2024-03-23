import pandas
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def img_to_arr_grey(data: pandas.DataFrame) -> [np.array, np.array]:
    x_train = np.empty((0, 96, 96), dtype=int)
    for i in tqdm(range(data.shape[0])):
        path = "./dataset/" + data.iloc[i, 0]
        img = cv2.imread(path)
        grey_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_train = np.append(x_train, [grey_s], axis=0)
    y_train = np.full(shape=(data.shape[0],), fill_value=data.iloc[0,1])
    return [x_train,y_train]

def main():
    # importing my data
    train_data = pd.read_csv('./dataset/train_data_labels.csv').drop(columns=['Unnamed: 0'])
    test_data = pd.read_csv('./dataset/test_data_labels.csv').drop(columns=['Unnamed: 0'])

    X_train, y_train = img_to_arr_grey(train_data)
    X_test, y_test = img_to_arr_grey(test_data)

    print("Finished Processing")
    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_test: ", y_test.shape)


if __name__ == "__main__":
    main()
