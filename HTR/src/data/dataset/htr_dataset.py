import glob
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

from src.data.image.img_preprocess import image_resize, centered_img
from src.data.text.read_txt_util import Text_Reader


class HTRDataset(Dataset):
    """

    """

    def __init__(self,
                 dir_data: str,
                 label_dir_img: str,
                 label_dir_label: str,
                 label_dir_boxes: str,
                 fixed_size,
                 width_divisor,
                 pad_left,
                 pad_right,
                 text_read: Text_Reader,
                 transforms: list = None,
                 ext_img: str = "png",
                 apply_noise: int = 0,
                 is_trainset=False):
        """
        """

        self.image_paths = []

        self.labels_str = []
        self.labels_ind = []
        self.id_item = []

        self.fixed_size = fixed_size
        self.transforms = transforms
        self.pad_left = pad_left
        self.pad_right = pad_right

        self.width_divisor = width_divisor

        self.apply_noise = apply_noise
        self.is_trainset = is_trainset

        dir_img = os.path.join(dir_data, label_dir_img)
        dir_label = os.path.join(dir_data, label_dir_label)

        # Images and label
        if ext_img == "pngjpg":
            files_img = glob.glob(dir_img + '/**/*.png', recursive=True)

            files_img.extend(glob.glob(dir_img + '/**/*.jpg', recursive=True))

        else:
            files_img = glob.glob(dir_img + '/**/*.' + ext_img, recursive=True)

        for one_file in files_img:
            # Get id file
            split_name = os.path.split(one_file)
            split_name = split_name[1].split(sep=".")  # Filename and extension
            id_file = split_name[0]

            path_label = os.path.join(dir_label, id_file + ".txt")

            if not os.path.isfile(path_label):
                print(one_file)
                print("label text associated doesn't exist. Data not loaded")
                continue

            label_str = text_read.read_text(path_label)
            label_ind = text_read.transcript_txt_to_index(label_str)

            self.labels_str.append(label_str)
            self.labels_ind.append(label_ind)

            self.id_item.append(id_file)

            self.image_paths.append(one_file)

    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        """
        paths_img = self.image_paths[idx]
        img = io.imread(paths_img, as_gray=True)  # , plugin='pil' add plugin for .tif image

        # Binarize img
        if img.dtype == bool:
            img = img.astype(int)
            img *= 255

        # Color image -> grayscale -> value between 0 and 1
        if img.dtype == float:
            img *= 255.0

        # grayscale image -> uint value 0 - 255

        # Resize and pad
        img = 1 - img.astype(np.float32) / 255.0

        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        # # https://github.com/georgeretsi/HTR-best-practices/blob/main/utils/transforms.py
        if self.is_trainset:
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

        img = centered_img(img, (fheight, fwidth), border_value=0.0)

        img = np.pad(img, ((0, 0), (self.pad_left, self.pad_right)), 'constant', constant_values=0)

        # Augmentation
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        imgs_shape = img.shape
        w_reduce = np.floor(imgs_shape[1] / self.width_divisor).astype(int)

        img_tensor = torch.as_tensor(img, dtype=torch.float32)

        if self.apply_noise == 1:
            if np.random.rand() < .33:
                img_tensor += torch.rand(img_tensor.size())

        img_tensor = img_tensor.unsqueeze(0)  # Add channel dim

        sample = {
            "ids": self.id_item[idx],

            "label_str": self.labels_str[idx],
            "label_ind": self.labels_ind[idx],

            "img": img_tensor,
            "w_reduce": w_reduce
        }

        return sample
