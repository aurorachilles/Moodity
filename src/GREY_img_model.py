import cv2
import numpy as np
import pandas as pd
from numpy import load
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPool2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def main():
    X_train = load("../data/dataset/npz_data/X_train.npz")["arr_0"]
    y_train = load("../data/dataset/npz_data/y_train.npz")["arr_0"]
    X_test = load("../data/dataset/npz_data/X_test.npz")["arr_0"]
    y_test = load("../data/dataset/npz_data/y_test.npz")["arr_0"]

    print("The X_train shape is: ", X_train.shape)
    print("The y_train shape is: ", y_train.shape)
    print("The X_test shape is: ", X_test.shape)
    print("The y_test shape is: ", y_test.shape)

    #preprocessing our data
    print("Reshaing our img data")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2] , 1)

    num_classes = pd.read_csv("../data/dataset/emotion_count.csv").shape[0]

    # onehot = pd.get_dummies(y_train)
    # target_labels = onehot.columns
    # dic = dict(zip(target_labels, range(num_classes)))

    label_enc = LabelEncoder()

    y_train_num = label_enc.fit_transform(y_train)
    y_test_num = label_enc.fit_transform(y_test)

    y_train = to_categorical(y_train_num, num_classes)
    y_test = to_categorical(y_test_num, num_classes)


    print("The X_train shape is: ", X_train.shape)
    print("The y_train shape is: ", y_train.shape)
    print("The X_test shape is: ", X_test.shape)
    print("The y_test shape is: ", y_test.shape)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255


    #Building our model


if __name__ == "__main__":
    main()