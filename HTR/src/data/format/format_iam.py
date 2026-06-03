import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

from pathlib import Path


# From git link to https://arxiv.org/pdf/2012.03868
def format_IAM_line(source_folder_split, source_folder, target_folder, tar_filename):
    """
    Format the IAM dataset at line level with the commonly used split (6,482 for train, 976 for validation and 2,915 for test)
    """

    line_folder_path = os.path.join(target_folder, "lines")

    tar_path = os.path.join(source_folder, tar_filename)
    if not os.path.isfile(tar_path):
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(target_folder, exist_ok=True)
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path)
    tar.close()

    set_names = ["train", "validation", "test"]

    for set_name in set_names:
        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)

        current_folder_img = os.path.join(current_folder, "img")
        os.makedirs(current_folder_img, exist_ok=True)

        current_folder_label = os.path.join(current_folder, "label")
        os.makedirs(current_folder_label, exist_ok=True)

        xml_path = os.path.join(source_folder_split, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]

            ids_img = [Path(path).stem for path in img_paths]

            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                img_name = ids_img[i] + ".png"
                label_name = ids_img[i] + ".txt"

                path_label = os.path.join(current_folder_label, label_name)
                with open(path_label, 'w', encoding="utf-8") as file:
                    file.write(label)

                new_path = os.path.join(current_folder_img, img_name)
                os.replace(img_paths[i], new_path)

    shutil.rmtree(line_folder_path)


if __name__ == "__main__":
    print("----IAM dataset----")

    # Split files for train, val and test
    source_folder_split = "../../../data/IAM/splits"  # "C:/Users/simcor/dev/data/IAM/split"  #"raw/IAM"

    # Direcotry with IAM data from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    source_folder = "C:/Users/simcor/dev/data/IAM/origin"

    # Where to save
    target_folder = "C:/Users/simcor/dev/data/IAM/test_format"

    tar_filename = "lines.tgz"

    format_IAM_line(source_folder_split, source_folder, target_folder, tar_filename)
