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



def evaluate_test(args, config_values, device, char_list, char_dict, ctc_loss_fn, text_read):
    # Model
    cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
    head_cfg = (256, 3) 
    width_divisor = 8

    model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char())
    # print("Initializing model weights kaiming")
    for p in model_reco.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p, nonlinearity="relu")


    if os.path.isfile(args.path_model):
        load_pretrained_model(args.path_model, model_reco, device)
    else: 
        print("No model found at " + args.path_model)

    model_reco = model_reco.to(device)

    test_db = HTRDataset(config_values["dataset_folder"],
                         fixed_size_img,
                         width_divisor,
                         args.pad_left,
                         args.pad_right,
                         text_read,
                         ext_img=config_values["extension_img"])

    # print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)
    # print()
    # print("--------Begin Test-----------")
    dict_result = evaluate_one_epoch_crnn(test_dataloader,
                                          model_reco,
                                          device,
                                          char_list,
                                          char_dict[BLANK_STR_TOKEN],
                                          ctc_loss_fn,
                                          text_read,
                                          args.compute_wer,
                                          args.use_wer_formula_for_cer)

    # dict_result["metrics_main"].print_cer_wer()

    out = dict_result["metrics_main"].get_cer()
    return out

if __name__ == '__main__':
    results_log = "../../logs/DomainAdaptation/results_crnn_baseline.txt"

    args = parser.parse_args()

    faulthandler.enable()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device :")
    print(device)
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))



    # Alphabet
    charset = CharsetToken("../../DATASETS/IAM/line_splits1/charset2.txt", use_blank=True)
    char_list = charset.get_charset_list()
    char_dict = charset.get_charset_dictionary()

    # Data
    fixed_size_img = (args.height_max, args.width_max)

    text_read = Text_Reader(args.read_txt_format, char_dict, args.add_space_before_after, args.space_value, args.filter_txt)

    # Pad img with black = 0
    c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
    collate_fn = c_collate_fn.collate_fn

    ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

    trained_on = {
        "IAM": {
            "trained_tag" : "016",
            "crnn_model": "016_2026-04-07_ID_16124657"
        },
        "Bentham": {
            "trained_tag" : "017",
            "crnn_model": "017_2026-04-07_ID_16124658"
        },
        "READ_2016": {
            "trained_tag" : "018",
            "crnn_model": "018_2026-04-07_ID_16124659"
        },
        "NorHandV3_72702": {
            "trained_tag" : "019",
            "crnn_model": "019_2026-04-07_ID_16124663"
        },
        "NorHandV3_331861": {
            "trained_tag" : "020",
            "crnn_model": "020_2026-04-07_ID_16124664"
        },
        "Riksarkivet_Bergskollegium": {
            "trained_tag" : "021",
            "crnn_model": "021_2026-04-07_ID_16124665"
        },
        "Riksarkivet_Goteborgs2": {
            "trained_tag" : "022",
            "crnn_model": "022_2026-04-11_ID_16194385"
        },
    }

    validate_on = {
        "IAM": {
            "dataset_sufix": "IAM",
            "dataset_folder": "../../DATASETS/IAM/line_splits1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/IAM/line_splits1/charset2.txt",
        },
        "Bentham": {
            "dataset_sufix": "Bentham",
            "dataset_folder": "../../DATASETS/Bentham/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/Bentham/polygon_lines_dataset1/charset2.txt",
        },
        "READ_2016": {
            "dataset_sufix": "READ_2016",
            "dataset_folder": "../../DATASETS/READ_2016/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/READ_2016/polygon_lines_dataset1/charset2.txt",
        },
        "NorHandV3_72702": {
            "dataset_sufix": "NorHandV3_72702",
            "dataset_folder": "../../DATASETS/NorHandV3_72702/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/NorHandV3_72702/polygon_lines_dataset1/charset2.txt",
        },
        "NorHandV3_331861": {
            "dataset_sufix": "NorHandV3_331861",
            "dataset_folder": "../../DATASETS/NorHandV3_331861/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/NorHandV3_331861/polygon_lines_dataset1/charset2.txt",
        },
        "Riksarkivet_Bergskollegium": {
            "dataset_sufix": "Riksarkivet_Bergskollegium",
            "dataset_folder": "../../DATASETS/Riksarkivet_Bergskollegium/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/Riksarkivet_Bergskollegium/polygon_lines_dataset1/charset2.txt",
        },
        "Riksarkivet_Goteborgs2": {
            "dataset_sufix": "Riksarkivet_Goteborgs2",
            "dataset_folder": "../../DATASETS/Riksarkivet_Goteborgs2/polygon_lines_dataset1",
            "extension_img": "png",
            "charset_file": "../../DATASETS/Riksarkivet_Goteborgs2/polygon_lines_dataset1/charset2.txt",
        },
    }

    for key_train, value_train in trained_on.items():
        print(f"Segmentation model Trained on: {key_train}")
        args.path_model = f"../../logs/CRNN_Center_loss/DomainAdaptation/{value_train['crnn_model']}/crnn_best.torch"

        for key_validate, value_validate in validate_on.items():
            print(f"Inference dataset: {key_validate}")
            config_values = {}
            # Paths
            config_values["dataset_folder"] = f"{value_validate['dataset_folder']}/test"
            config_values["extension_img"] = value_validate["extension_img"]
            # config_values["charset_file"] = value_validate["charset_file"]

            out_cer = evaluate_test(args, config_values, device, char_list, char_dict, ctc_loss_fn, text_read)
            out_cer_presentage = f"{100 * out_cer:.2f}"
            print(out_cer_presentage)

            text = "Baseline"
            with open(results_log, "a") as f:
                f.write(f"{key_train};{value_train['trained_tag']};{value_validate['dataset_sufix']};{out_cer_presentage}\n")
        
