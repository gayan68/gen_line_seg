import glob
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

from src.data.image.img_preprocess import image_resize, centered_img
from src.data.text.read_txt_util import Text_Reader
import lmdb
import io
from PIL import Image


class HTRDataset(Dataset):
    """

    """

    def __init__(self,
                 dir_data: str,
                 fixed_size,
                 width_divisor,
                 pad_left,
                 pad_right,
                 text_read: Text_Reader,
                 transforms: list = None,
                 apply_noise: int = 0,
                 is_trainset=False):
        """
        """

        self.keys = []

        self.text_read = text_read

        self.fixed_size = fixed_size
        self.transforms = transforms
        self.pad_left = pad_left
        self.pad_right = pad_right

        self.width_divisor = width_divisor

        self.apply_noise = apply_noise
        self.is_trainset = is_trainset

        self.lmdb_path = os.path.join(dir_data, "lmdb")
        self.env = None   # IMPORTANT: do NOT open LMDB here

        # Read keys ONLY using a temporary env
        tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        with tmp_env.begin() as txn:
            cursor = txn.cursor()  # create a cursor to iterate

            for key, value in cursor:
                key_str = key.decode()  # convert bytes -> string
                if key_str.startswith("image-"):
                    self.keys.append(key_str[6:])
        tmp_env.close()



    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        return len(self.keys)

    def __getitem__(self, idx):
        """
        """
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        with self.env.begin() as txn:
            img_key = f"image-{self.keys[idx]}".encode()
            gt_key = f"label-{self.keys[idx]}".encode()
            line_id = self.keys[idx]

            img_bin = txn.get(img_key)
            label_str = txn.get(gt_key).decode() # string


        img = Image.open(io.BytesIO(img_bin)).convert("L")  # Convert to grayscale
        label_str = self.text_read.read_text2(label_str)
        labels_ind = self.text_read.transcript_txt_to_index(label_str)

        img = np.array(img)

        # Binarize img
        if img.dtype == bool:
            img = img.astype(int)
            img *= 255
            print("Binarized img detected. Converted to uint8")

        if np.max(img) <= 1:
            img = (img * 255).astype(int)

        # print(np.min(img), np.max(img))
        # print("img shape:", img.shape)


        # Resize and pad
        img = 1 - img.astype(np.float32) / 255.0
        img = np.clip(img, 0, 1)

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
            "ids": line_id,

            "label_str": label_str,
            "label_ind": labels_ind,

            "img": img_tensor,
            "w_reduce": w_reduce
        }


        return sample