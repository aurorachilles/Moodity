import pandas as pd
import sys
import numpy as np
import cv2


def main():
    data = pd.read_csv('./dataset/labels.csv').drop(columns=['Unnamed: 0'])
    path = "./dataset/"+data.iloc[2, 0]
    print(data)
    img = cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat_gray = gray_image.flatten()
    print(data.iloc[0:, 2].mean())





if __name__ == "__main__":
    main()
