import pandas
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def img_to_arr_grey(data: pandas.DataFrame) -> [np.array, np.array]:
    x_ = np.empty((0, 96, 96), dtype=int)
    y_ = np.empty((0,), dtype=str)
    for i in tqdm(range(data.shape[0])):
        path = "./dataset/" + data.iloc[i, 0]
        img = cv2.imread(path)
        grey_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_ = np.append(x_, [grey_s], axis=0)
        y_ = np.append(y_, data.iloc[i, 1])
    return [x_,y_]

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

    y_train_resize = y_train.reshape(-1, 1, 1)
    y_test_resize = y_test.reshape(-1, 1, 1)

    np.savez("./dataset/npz_data/X_train.npz", X_train)
    np.savez("./dataset/npz_data/y_train.npz", y_train)
    np.savez("./dataset/npz_data/X_test.npz", X_test)
    np.savez("./dataset/npz_data/y_test.npz", y_test)



if __name__ == "__main__":
    main()
