import argparse
import faulthandler
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data.batch.collate_batch import CollateImageLabelHTR
from src.data.dataset.htr_dataset_lmdb import HTRDataset
from src.data.text.charset_token import CharsetToken
from src.data.text.common_token_txt import CTC_PAD, BLANK_STR_TOKEN
from src.data.text.read_txt_util import READ_TEXT_FORMAT, SPACE_VALUE, FILTER_TXT, Text_Reader
from src.evaluate.evaluate_crnn_one_epoch import evaluate_one_epoch_crnn
from src.model.crnn import CRNN
from src.model.models_utils import load_pretrained_model
import json

parser = argparse.ArgumentParser()

parser.add_argument("config_file")

parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument("--path_model", default="", type=str)
parser.add_argument('--height_max', default=128, type=int)
parser.add_argument('--width_max', default=1700, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)

# Dataset text specificity
parser.add_argument('--read_txt_format', type=lambda tw: READ_TEXT_FORMAT[tw], choices=list(READ_TEXT_FORMAT),
                    default=READ_TEXT_FORMAT.RAW)
parser.add_argument('--add_space_before_after', default=1, type=int)  # 1 = activate
parser.add_argument('--space_value', type=lambda tw: SPACE_VALUE[tw], choices=list(SPACE_VALUE),
                    default=SPACE_VALUE.RAW)
parser.add_argument('--filter_txt', type=lambda tw: FILTER_TXT[tw], choices=list(FILTER_TXT), default=FILTER_TXT.NO)
parser.add_argument('--compute_wer', default=1, type=int)
parser.add_argument('--use_wer_formula_for_cer', default=0, type=int)

parser.add_argument('--val_data_exist', default=1, type=int)
parser.add_argument('--test_data_exist', default=1, type=int)
parser.add_argument("--label_dir_img", default="Images", type=str)
parser.add_argument("--label_dir_label", default="gt_text", type=str)
parser.add_argument("--label_dir_boxes", default="gt_boxes", type=str)
parser.add_argument("--split", default="Validation", type=str)
print("===============================================================================")

begin = time.time()
args = parser.parse_args()
print(args)

faulthandler.enable()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

config_values = {}

with open(args.config_file, "r") as fp:
    config_values = json.load(fp)

# Paths
dataset_folder = config_values["dataset_folder"]

if args.val_data_exist == 1:
    directory_split = os.path.join(dataset_folder, "val")

ext_img = config_values["extension_img"]
charset_file = config_values["charset_file"]

# Alphabet
charset = CharsetToken(charset_file, use_blank=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

# Model
# cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
# head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
# width_divisor = 8

cnn_cfg = config_values["cnn_cfg"]
head_cfg = config_values["head_cfg"]  # (hidden dimension, num_layers blstm)
width_divisor = config_values["width_divisor"]

model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char())

# Data
fixed_size_img = (args.height_max, args.width_max)

text_read = Text_Reader(args.read_txt_format, char_dict, args.add_space_before_after, args.space_value, args.filter_txt)

if args.val_data_exist == 1:
    val_db = HTRDataset(directory_split,
                        fixed_size_img,
                        width_divisor,
                        args.pad_left,
                        args.pad_right,
                        text_read,
                        ext_img=ext_img)

    print('Nb samples val {}:'.format(len(val_db)))

# Pad img with black = 0
c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
collate_fn = c_collate_fn.collate_fn

if args.val_data_exist == 1:
    val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                collate_fn=collate_fn, shuffle=False)

print("Initializing model weights kaiming")
for p in model_reco.parameters():
    if p.dim() > 1:
        torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

if os.path.isfile(args.path_model):
    load_pretrained_model(args.path_model, model_reco, device)

print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

print_summary = ""

ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

ceneters_value = []
compute_loss_reg = False

begin_eval = time.time()

if args.val_data_exist == 1:
    print("--------------Evaluate-------------------------------")
    dict_result = evaluate_one_epoch_crnn(val_dataloader,
                                          model_reco,
                                          device,
                                          char_list,
                                          char_dict[BLANK_STR_TOKEN],
                                          ctc_loss_fn,
                                          text_read,
                                          args.compute_wer,
                                          args.use_wer_formula_for_cer)

    dict_result["metrics_main"].print_cer_wer()
    print_summary += "Validation split \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"

if args.test_data_exist == 1:
    directory_test = os.path.join(dataset_folder, "test")

    test_db = HTRDataset(directory_test,
                         fixed_size_img,
                         width_divisor,
                         args.pad_left,
                         args.pad_right,
                         text_read,
                         ext_img=ext_img)

    print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)
    print()
    print("--------Begin Test-----------")
    dict_result = evaluate_one_epoch_crnn(test_dataloader,
                                          model_reco,
                                          device,
                                          char_list,
                                          char_dict[BLANK_STR_TOKEN],
                                          ctc_loss_fn,
                                          text_read,
                                          args.compute_wer,
                                          args.use_wer_formula_for_cer)

    dict_result["metrics_main"].print_cer_wer()

    print_summary += "Test split \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"

end_eval = time.time()
print("Time all (s): " + str((end_eval - begin_eval)))
print("End training")
print()
print("Summary:")
print(print_summary)
