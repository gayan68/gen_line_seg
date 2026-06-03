# CRNN Model for Handwritten Text Recognition (HTR)

## Overview

This repository contains the implementation and training configuration of the **Convolutional Recurrent Neural Network (CRNN)** model used in the paper:

> **Generalization of Text Line Segmentation for HTR in Historical Documents**

The CRNN model serves as the Handwritten Text Recognition (HTR) backbone for evaluating the impact of text line segmentation quality on recognition performance across historical document collections.

The architecture combines:

* **Convolutional Neural Networks (CNNs)** for visual feature extraction from text line images.
* **Recurrent Neural Networks (RNNs)** for sequence modeling.
* **Connectionist Temporal Classification (CTC)** loss for alignment-free transcription learning.

---

## Requirements

* Python 3.x
* PyTorch
* LMDB dataset format
* Additional dependencies listed in `requirements.txt`

---

## Training

A typical training command is shown below:

```bash
python -u train_crnn.py \
  "configuration/config_lmdb_NorHand_331861.json" \
  "../../../logs/HTR/NorHand_331861" \
  --num_workers 4 \
  --batch_size 32 \
  --height_max 128 \
  --width_max 1024 \
  --nb_epochs_max 1500 \
  --lr_drop_patience 100 \
  --early_stop_patience 200 \
  --use_regularization 0 \
  --epoch_start_regularization 20000
```

### Training Parameters

| Parameter                                       | Description                                            |
| ----------------------------------------------- | ------------------------------------------------------ |
| `configuration/config_lmdb_NorHand_331861.json` | Dataset and model configuration file                   |
| `../../../logs/HTR/NorHand_331861`              | Output directory for logs and checkpoints              |
| `--num_workers 4`                               | Number of data loading workers                         |
| `--batch_size 32`                               | Training batch size                                    |
| `--height_max 128`                              | Maximum input image height                             |
| `--width_max 1024`                              | Maximum input image width                              |
| `--nb_epochs_max 1500`                          | Maximum number of training epochs                      |
| `--lr_drop_patience 100`                        | Epochs to wait before reducing learning rate           |
| `--early_stop_patience 200`                     | Early stopping patience                                |
| `--use_regularization 0`                        | Disable regularization during training                 |
| `--epoch_start_regularization 20000`            | Epoch at which regularization would begin (if enabled) |

---

## Dataset Format

Training data should be provided in LMDB format as specified in the configuration JSON file. The configuration file defines:

* Dataset locations
* Character set
* Model parameters
* Training and validation splits
* Data preprocessing settings

---

## Outputs

During training, the following artifacts are generated:

* Model checkpoints
* Training logs
* Validation metrics
* Best-performing model weights

These outputs are stored in the specified logging directory.

---


