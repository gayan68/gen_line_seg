import torch
import os
import json
import argparse
from src.data.text.read_txt_util import Text_Reader
from torch.utils.data import DataLoader
from src.data.dataset.htr_dataset_lmdb import HTRDataset
from src.model.crnn import CRNN
from src.model.models_utils import load_pretrained_model
from src.data.batch.collate_batch import CollateImageLabelHTR
from src.data.text.common_token_txt import CTC_PAD, BLANK_STR_TOKEN
from src.data.text.read_txt_util import READ_TEXT_FORMAT, SPACE_VALUE, FILTER_TXT, Text_Reader
from src.data.text.charset_token import CharsetToken

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
parser.add_argument("--path_model", default="", type=str)
parser.add_argument('--height_max', default=128, type=int)
parser.add_argument('--width_max', default=1700, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)
parser.add_argument('--read_txt_format', type=lambda tw: READ_TEXT_FORMAT[tw], choices=list(READ_TEXT_FORMAT),
                    default=READ_TEXT_FORMAT.RAW)
parser.add_argument('--add_space_before_after', default=1, type=int)  # 1 = activate
parser.add_argument('--space_value', type=lambda tw: SPACE_VALUE[tw], choices=list(SPACE_VALUE),
                    default=SPACE_VALUE.RAW)
parser.add_argument('--filter_txt', type=lambda tw: FILTER_TXT[tw], choices=list(FILTER_TXT), default=FILTER_TXT.NO)
parser.add_argument("--save_dir", default="", type=str)

args = parser.parse_args()

config_values = {}
with open(args.config_file, "r") as fp:
    config_values = json.load(fp)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict(data_loader, model, device, char_list, token_blank, text_read):

    model.eval()
    page_name = ""
    pred_output = {}
    onepage = []

    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]
            line_id = batch_data["ids"]

            y_enc = batch_data["label_ind"].to(device)
            y_len_enc = batch_data["label_ind_length"]

            y_gt_txt = batch_data["label_str"]
        
            # Remove text padding
            y_gt_txt = [text_read.remove_space_before_after_one_item(t) for t in y_gt_txt]

            nb_item_batch = x.shape[0]

            y, _, _ = model(x)
            output, aux_output = y

            # Main head
            output_log = torch.nn.functional.log_softmax(output, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(output_log)]
            predictions_text = [text_read.ctc_best_path_one(p, char_list, token_blank) for p in top]

            predictions_text = [text_read.remove_space_before_after_one_item(t) for t in predictions_text]  # Remove text padding

            if line_id[0][:-3] != page_name:
                if len(onepage)>0 :
                    pred_output[page_name] = onepage
                onepage = []
            onepage.append(predictions_text[0])
            page_name = line_id[0][:-3]

        pred_output[page_name] = onepage

        return pred_output


if __name__ == '__main__':
                                  
    dataset_folder = config_values["dataset_folder"]
    directory_test = os.path.join(dataset_folder, "test")
    fixed_size_img = (args.height_max, args.width_max)

    ext_img = config_values["extension_img"]
    charset_file = config_values["charset_file"]

    # Alphabet
    charset = CharsetToken(charset_file, use_blank=True)
    char_list = charset.get_charset_list()
    char_dict = charset.get_charset_dictionary()

    # Model
    cnn_cfg = config_values["cnn_cfg"]
    head_cfg = config_values["head_cfg"]  # (hidden dimension, num_layers blstm)
    width_divisor = config_values["width_divisor"]

    model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char())

    if os.path.isfile(args.path_model):
        load_pretrained_model(args.path_model, model_reco, device)

    text_read = Text_Reader(args.read_txt_format, char_dict, args.add_space_before_after, args.space_value, args.filter_txt)


    test_db = HTRDataset(directory_test,
                            fixed_size_img,
                            width_divisor,
                            args.pad_left,
                            args.pad_right,
                            text_read,
                            ext_img=ext_img)
    # Pad img with black = 0
    c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
    collate_fn = c_collate_fn.collate_fn

    test_dataloader = DataLoader(test_db, num_workers=1, batch_size=1, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)
    

    pred_out = predict(test_dataloader, model_reco, device, char_list, char_dict[BLANK_STR_TOKEN], text_read)

    os.makedirs(args.save_dir, exist_ok=True)
    for key, value in pred_out.items():
        print(key)
        with open(f"{args.save_dir}/{key}.txt", "w") as f:
            for text_line in value:
                print(text_line)
                f.write(text_line + "\n")