import pandas
import pandas as pd
import cv2
from tqdm import tqdm


def img_to_arr_grey(data: pandas.DataFrame) -> pandas.DataFrame:
    new_data = pd.DataFrame(data)
    new_data['img_vector'] = ""

    for i in tqdm(range(data.shape[0])):
        path = "./dataset/" + data.iloc[i, 0]
        img = cv2.imread(path)
        grey_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normal = grey_s.flatten()
        new_data.iloc[i, 3] = (" ".join(map(str, normal)))
    return new_data


def main():
    # importing my data
    data = pd.read_csv('./dataset/labels.csv').drop(columns=['Unnamed: 0'])
    data = img_to_arr_grey(data)
    print("Finished Processing")
    data.to_csv('./dataset/new_labels.csv')


if __name__ == "__main__":
    main()
