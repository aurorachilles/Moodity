import sys
import os
import pandas as pd
from tqdm import tqdm


def sub_dirs_calc():
    data = pd.read_csv('./dataset/new_labels.csv')
    cols = list(data['label'].unique())
    return cols


def create_test_split_folder():
    print("Reading metadata, please wait")
    main_directory = "./dataset/"
    folder_name = "train_test"
    head_dirs = ['train', 'test']
    sub_dirs = sub_dirs_calc()
    print("Reading done!")
    print("Starting creation of dirs")
    try:
        # creating the train test folder that will hold the data
        print("Creating directories for the dataset")
        os.makedirs(os.path.join(main_directory, folder_name))
        for head_dir in tqdm(head_dirs):
            print("creating main dir:", head_dir)
            os.makedirs(os.path.join(main_directory, folder_name, head_dir), exist_ok=True)
            for sub_dir in sub_dirs:
                print("creating sub dir:", sub_dir)
                os.makedirs(os.path.join(main_directory, folder_name, head_dir, sub_dir), exist_ok=True)

    except OSError:
        print("Creation of the directory %s failed at" % folder_name, "because folder already exists")
    print("Finished creating Directories")


if __name__ == "__main__":
    create_test_split_folder()
    # print(sys.argv)
    print("Folders created successfully! in Data folder.")
    # sys.exit(0)
