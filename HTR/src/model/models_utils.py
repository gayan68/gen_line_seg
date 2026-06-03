import os
import torch


def load_pretrained_model(pretrained_model_file, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        checkpoint = torch.load(pretrained_model_file, map_location=device)

        pretrained_dict = checkpoint
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict:
                if model_dict[k].shape == pretrained_dict[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                if "top.rec" in k:
                    print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)


def load_pretrained_model_LSTM(pretrained_model_file_cnn, pretrained_model_file_lstm, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file_cnn + ")...")
    if os.path.isfile(pretrained_model_file_cnn) and os.path.isfile(pretrained_model_file_lstm):
        checkpoint_cnn = torch.load(pretrained_model_file_cnn, map_location=device)
        checkpoint_lstm = torch.load(pretrained_model_file_lstm, map_location=device)

        pretrained_dict_cnn = checkpoint_cnn
        pretrained_dict_lstm = checkpoint_lstm
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict_cnn:
                if "features.features" in k or "top.cnn" in k:
                    if model_dict[k].shape == pretrained_dict_cnn[key].shape:
                        pretrained_keys.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

            if key in pretrained_dict_lstm:
                if "top.rec" in k or "top.fnl" in k:
                    if model_dict[k].shape == pretrained_dict_lstm[key].shape:
                        pretrained_keys.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if "top.rec" in k or "top.fnl" in k:
                model_dict[k] = pretrained_dict_lstm[key]
                if print_load_ok:
                    print(f"Loading LSTM weight: {k}")
            else:
                model_dict[k] = pretrained_dict_cnn[key]
                if print_load_ok:
                    print(f"Loading CNN weight: {k}")
            

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict_cnn[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file_cnn + " OR " + pretrained_model_file_lstm + " ...")

    model.to(device)