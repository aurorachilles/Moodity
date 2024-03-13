import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

def main():
    # importing data
    data = pd.read_csv('./dataset/new_labels.csv').drop(columns=['Unnamed: 0'])
    mat_data = data.iloc[1, 3].split()

    # matrix for making image
    matrix = np.zeros((96, 96))

    for i in range(96 * 96):
        # print(mat_data[i])
        matrix[i // 96][i % 96] = mat_data[i]

    save_img = Image.fromarray(matrix)

    img_sub_folder = data.iloc[1, 1]


if __name__ == "__main__":
    main()
