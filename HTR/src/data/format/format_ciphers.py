import os
import glob
import random
import shutil
from pathlib import Path


def split_train_val(origin_dir, save_dir, ratio_train, ext_img):
    """divided one dataset into training and validation split

    Division is NOT random"""

    files_img = glob.glob(origin_dir + '/**/*.' + ext_img, recursive=True)
    random.shuffle(files_img)

    nb_train = int(ratio_train * len(files_img))

    # Train
    save_dir_train = os.path.join(save_dir, "train")

    save_img = os.path.join(save_dir_train, "img")
    save_label = os.path.join(save_dir_train, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    for one_img_path in files_img[:nb_train]:

        id_sample = Path(one_img_path).stem

        path_to = os.path.join(save_img, id_sample + "." + ext_img)
        shutil.copyfile(one_img_path, path_to)

        path_label_origin = os.path.join(origin_dir, "txt", id_sample + ".txt")

        path_to = os.path.join(save_label, id_sample + ".txt")
        shutil.copyfile(path_label_origin, path_to)

    # Validation
    save_dir_val = os.path.join(save_dir, "validation")

    save_img = os.path.join(save_dir_val, "img")
    save_label = os.path.join(save_dir_val, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    for one_img_path in files_img[nb_train:]:

        id_sample = Path(one_img_path).stem

        path_to = os.path.join(save_img, id_sample + "." + ext_img)
        shutil.copyfile(one_img_path, path_to)

        path_label_origin = os.path.join(origin_dir, "txt", id_sample + ".txt")

        path_to = os.path.join(save_label, id_sample + ".txt")
        shutil.copyfile(path_label_origin, path_to)


if __name__ == "__main__":
    print("----Cipher task 1----")
    # Link to download the data: https://rrc.cvc.uab.es/?ch=27&com=downloads

    # Direcotry with extracted train data
    source_folder = "C:/Users/simcor/dev/data/Cipher/test_format/task1/"
    save_dir = "C:/Users/simcor/dev/data/Cipher/test_format/Split_train_val/"

    # split data into 2 splits
    split_train_val(source_folder, save_dir, ratio_train=0.8, ext_img="png")
