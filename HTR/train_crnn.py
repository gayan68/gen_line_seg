import argparse
import faulthandler
import json
import os
import time
import threading

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.clusters.cluster_helper import compute_center_coordinates
from src.data.augmentation.get_augmentation_transform import get_augmentation_img
from src.data.batch.collate_batch import CollateImageLabelHTR
from src.data.dataset.htr_dataset_lmdb import HTRDataset
from src.data.text.charset_token import CharsetToken
from src.data.text.common_token_txt import CTC_PAD, BLANK_STR_TOKEN
from src.data.text.read_txt_util import READ_TEXT_FORMAT, SPACE_VALUE, FILTER_TXT, Text_Reader
from src.evaluate.evaluate_crnn_one_epoch import evaluate_one_epoch_crnn
from src.model.crnn import CRNN
from src.model.models_utils import load_pretrained_model
from src.train.train_crnn_one_epoch import train_crnn_reg_one_epoch, train_crnn_one_epoch
import wandb

import numpy as np
from PIL import Image

class GPUHeartbeat(threading.Thread):
    def __init__(self, device, interval=2.0):
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            # Small GPU op to keep utilization > 0
            _ = torch.randn(256, 256, device=self.device) @ \
                torch.randn(256, 256, device=self.device)
            torch.cuda.synchronize(self.device)
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")
    parser.add_argument("log_dir")

    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--nb_epochs_max', default=900, type=int)
    parser.add_argument("--path_model", default="", type=str)
    parser.add_argument("--path_optimizer", default="", help="", type=str)
    parser.add_argument('--height_max', default=128, type=int)
    parser.add_argument('--width_max', default=1700, type=int)
    parser.add_argument('--pad_left', default=64, type=int)
    parser.add_argument('--pad_right', default=64, type=int)

    parser.add_argument('--LR_drop', default=False, type=bool)
    parser.add_argument('--lr_decay_1', default=10, type=float)
    parser.add_argument('--lr_drop_patience', default=50, type=int)
    parser.add_argument('--early_stop_patience', default=50, type=int)

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
    parser.add_argument("--split", default="Validation", type=str)

    # Regularization
    parser.add_argument('--use_regularization', default=1, type=int)
    parser.add_argument('--epoch_start_regularization', default=650, type=int)
    parser.add_argument('--weight_loss_regularization_ok', default=0.55, type=float)

    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument("--wandb_proj_name", default="", help="", type=str)
    parser.add_argument('--save_intermediate', default=False, type=bool)

    parser.add_argument('--train_strategy', default="crnn", type=str)

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
    os.makedirs(args.log_dir, exist_ok=True)
    directory_log = args.log_dir
    dataset_folder = config_values["dataset_folder"]

    directory_train = os.path.join(dataset_folder, "train")

    if args.val_data_exist == 1:
        directory_val = os.path.join(dataset_folder, "val")

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

    # cnn_cfg = args.cnn_cfg
    # head_cfg = tuple(args.head_cfg) # (hidden dimension, num_layers blstm)
    # width_divisor = args.width_divisor

    model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char())

    # Data
    fixed_size_img = (args.height_max, args.width_max)

    aug_transforms = get_augmentation_img()

    text_read = Text_Reader(args.read_txt_format, char_dict, args.add_space_before_after, args.space_value, args.filter_txt)
    print(f"pad_left {args.pad_left}, pad_right {args.pad_right}")
    train_db = HTRDataset(directory_train,
                        fixed_size_img,
                        width_divisor,
                        args.pad_left,
                        args.pad_right,
                        text_read,
                        aug_transforms,
                        apply_noise=1,
                        is_trainset=True)
    print('Nb samples train {}:'.format(len(train_db)))

    if args.val_data_exist == 1:
        val_db = HTRDataset(directory_val,
                            fixed_size_img,
                            width_divisor,
                            args.pad_left,
                            args.pad_right,
                            text_read)
        print('Nb samples val {}:'.format(len(val_db)))

    ##### Debug a Sample Image #####

    # print(train_db[0])
    # sample = train_db[0]  # your dict
    # img_tensor = sample["img"]  # shape [1, H, W]
    # img = img_tensor.squeeze(0)
    # img_np = img.numpy()
    # img_np = (img_np * 255).astype(np.uint8)
    # img_pil = Image.fromarray(img_np, mode="L")  # "L" = grayscale
    # img_pil.save("debug_output.jpg")
    # print("Saved as debug_output.jpg")
    # GAYAN
    ##### End Debug #####

    # Pad img with black = 0
    c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
    collate_fn = c_collate_fn.collate_fn

    train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                collate_fn=collate_fn, shuffle=True)

    if args.val_data_exist == 1:
        val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)

    # Init model
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

    optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

    if os.path.isfile(args.path_optimizer):
        try:
            checkpoint = torch.load(args.path_optimizer, map_location=device)
            optimizer.load_state_dict(checkpoint)
            print("Load optimizer")
        except:
            print("Error load optimizer")
            optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

    best_cer = 1.0
    best_epoch = 0

    path_save_model_best = os.path.join(directory_log, "crnn_best.torch")
    path_save_model_last = os.path.join(directory_log, "crnn_last.torch")

    path_save_optimizer_best = os.path.join(directory_log, "optimizer_best.torch")
    path_save_optimizer_last = os.path.join(directory_log, "optimizer_last.torch")

    lr = args.learning_rate
    # For center loss
    index_class_to_filter = [char_dict["<BLANK>"]]

    if args.read_txt_format == READ_TEXT_FORMAT.RAW:
        index_class_to_filter.append(char_dict[" "])
    elif args.read_txt_format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
        if "<SPACE>" in char_dict:
            index_class_to_filter.append(char_dict["<SPACE>"])

    loss_reg = torch.nn.MSELoss(reduction="mean")

    conf_reg = {
        "index_class_to_filter": index_class_to_filter,
        "loss_reg": loss_reg,
        "weight_loss_regularization_ok": args.weight_loss_regularization_ok
    }

    ceneters_value = []
    compute_loss_reg = False

    ### --- Initialize Wandb ---
    print(f"self.args.wandb: {args.wandb}")

    if args.wandb:
        wandb.init(
            project=args.wandb_proj_name,
            config={
            "architecture": "CRNN",
            }
        )
        print("run name WANDB : " + str(wandb.run.name))   

    # -------------------------
    # Start GPU heartbeat
    # -------------------------
    heartbeat = GPUHeartbeat(device=device, interval=2.0)
    heartbeat.start()

    begin_train = time.time()
    # Training
    lr = args.learning_rate
    val_stable_count = 0
    epoch2 = 0
    LR_drop = args.LR_drop

    print(f"LR_drop: {LR_drop}")

    for epoch in range(0, args.nb_epochs_max):
        begin_time_epoch = time.time()
        print('EPOCH {}:'.format(epoch))

        if (val_stable_count >= args.lr_drop_patience) and (LR_drop == False):
            lr = args.learning_rate / args.lr_decay_1
            LR_drop = True
            val_stable_count = 0
            best_cer = 1.0
            print(f'LR Drop at: {epoch}')

        if LR_drop:
            epoch2 += 1

        # Learning rate values
        # if epoch < args.milestones_lr_1:
        #     lr = args.learning_rate
        # # decay
        # else:
        #     lr = args.learning_rate / args.lr_decay_1

        for g in optimizer.param_groups:
            g['lr'] = lr
        print("lr:" + str(lr))

        # Compute clusters coordinates if activate
        if args.use_regularization == 1 and LR_drop:
            if epoch2 >= args.epoch_start_regularization:
                print("Compute prototype")
                compute_loss_reg = True

                ceneters_value = compute_center_coordinates(train_dataloader,
                                                            model_reco,
                                                            device,
                                                            char_list,
                                                            char_dict["<BLANK>"],
                                                            index_class_to_filter,
                                                            text_read)

        # Training
        if compute_loss_reg:
            dict_losses = train_crnn_reg_one_epoch(train_dataloader,
                                                optimizer,
                                                model_reco,
                                                device,
                                                ctc_loss_fn,
                                                conf_reg,
                                                ceneters_value,
                                                char_list,
                                                char_dict[BLANK_STR_TOKEN],
                                                text_read)
        else:
            dict_losses = train_crnn_one_epoch(train_dataloader,
                                            optimizer,
                                            model_reco,
                                            device,
                                            ctc_loss_fn,
                                            args)

        print('train_loss_main {}'.format(dict_losses["loss_main"]))

        if compute_loss_reg:
            print('train_loss_reg_epoch {}'.format(dict_losses["loss_reg_epoch"]))

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

            if args.wandb :
                wandb.log({
                    "training_loss": dict_losses["loss_main"],
                    "validation_loss" : dict_result["metrics_main"].get_loss(),
                    "Val_CER": dict_result["metrics_main"].get_cer(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                    })
            
            # Save model
            if dict_result["metrics_main"].get_cer() < best_cer:
                best_cer = dict_result["metrics_main"].get_cer()
                best_epoch = epoch
                val_stable_count = 0
                print("Best cer final, save model.")

                torch.save(model_reco.state_dict(), path_save_model_best)
                torch.save(optimizer.state_dict(), path_save_optimizer_best)
            else:
                val_stable_count += 1


            # Save important points
            if epoch2 in [args.epoch_start_regularization - 1]:
                print(f"Model Save at: {epoch} | CER: {dict_result['metrics_main'].get_cer()}")

                intermediate_path_model = os.path.join(directory_log, f"crnn_epoch_{epoch}.torch")
                intermediate_path_optimizer = os.path.join(directory_log, f"optimizer_epoch_{epoch}.torch")
                torch.save(model_reco.state_dict(), intermediate_path_model)
                torch.save(optimizer.state_dict(), intermediate_path_optimizer)
            
            # Save intermediate model
            if (epoch % 50 ==0) and (args.save_intermediate):
                print(f"CER at epoch {epoch}: {dict_result['metrics_main'].get_cer()}")

                intermediate_path_model = os.path.join(directory_log, f"crnn_intermediate_epoch_{epoch}.torch")
                intermediate_path_optimizer = os.path.join(directory_log, f"optimizer_intermediate_epoch_{epoch}.torch")
                torch.save(model_reco.state_dict(), intermediate_path_model)
                torch.save(optimizer.state_dict(), intermediate_path_optimizer)

            if (val_stable_count >= args.early_stop_patience) and LR_drop :
                print('Early Stopping at: %d. Best epoch is: %d' % (epoch, best_epoch))
                torch.save(model_reco.state_dict(), os.path.join(directory_log, f"crnn_last_{epoch}.torch"))
                torch.save(optimizer.state_dict(), os.path.join(directory_log, f"optimizer_last_{epoch}.torch"))
                if args.wandb:
                    wandb.log({"Early Stopping at": epoch, "Best Epoch": best_epoch })
                    if args.test_data_exist != 1:
                        wandb.finish()
                break

        # # Compute clusters coordinates if activate
        # if args.use_regularization == 1:
        #     if epoch >= args.epoch_start_regularization:
        #         print("Compute prototype")
        #         compute_loss_reg = True

        #         ceneters_value = compute_center_coordinates(train_dataloader,
        #                                                     model_reco,
        #                                                     device,
        #                                                     char_list,
        #                                                     char_dict["<BLANK>"],
        #                                                     index_class_to_filter,
        #                                                     text_read)

        end_time_epoch = time.time()
        print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
        print("")

        torch.save(model_reco.state_dict(), path_save_model_last)
        torch.save(optimizer.state_dict(), path_save_optimizer_last)

    end_train = time.time()
    print("best_epoch: " + str(best_epoch))
    print("best_cer val: " + str(best_cer))
    print("Time all (s): " + str((end_train - begin_train)))
    print("End training")

    print_summary += "best_epoch: " + str(best_epoch) + "\n"
    print_summary += "best_cer val: "
    print_summary += f"{100 * best_cer:.2f}% \n"
    print_summary += "\n"

    heartbeat.stop()

    if args.test_data_exist == 1:
        directory_test = os.path.join(dataset_folder, "test")

        test_db = HTRDataset(directory_test,
                            fixed_size_img,
                            width_divisor,
                            args.pad_left,
                            args.pad_right,
                            text_read)

        print('Nb samples test {}:'.format(len(test_db)))

        test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)
        print()
        print("--------Begin Testing last-----------")
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

        print_summary += "Testing last \n"
        str_cer_wer = dict_result["metrics_main"].str_cer_wer()
        print_summary += str_cer_wer + "\n"
        final_test_cer = dict_result["metrics_main"].get_cer()

        if args.val_data_exist == 1:
            print("--------Begin Testing best cer val-----------")
            # Load best model
            if os.path.isfile(path_save_model_best):
                load_pretrained_model(path_save_model_best, model_reco, device, print_load_ok=False)

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

            print_summary += "\n"
            print_summary += "Testing best val cer \n"
            str_cer_wer = dict_result["metrics_main"].str_cer_wer()
            print_summary += str_cer_wer + "\n"
            final_val_cer = dict_result["metrics_main"].get_cer()

        if args.wandb :
            wandb.log({
                "Final_Val_CER": final_val_cer,
                "Final_Test_CER": final_test_cer
                })
            wandb.finish()

    print()
    print("Summary:")
    print(print_summary)

if __name__ == "__main__":
    main()