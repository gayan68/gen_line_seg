import editdistance
import torch

from src.clusters.cluster_helper import groupe_features_per_class, compute_center_loss_k1


def train_crnn_one_epoch(training_loader,
                         optimizer,
                         model,
                         device,
                         ctc_loss,
                         args,
                         weight_loss_shortcut=0.1):
    loss_main_epoch = 0
    loss_shortcut_epoch = 0

    model.train()

    for index_batch, batch_data in enumerate(training_loader):
        optimizer.zero_grad()

        x = batch_data["imgs"].to(device)
        x_reduced_len = batch_data["w_reduce"]

        y_enc = batch_data["label_ind"].to(device)
        y_len_enc = batch_data["label_ind_length"]
  
        y, _, _ = model(x)

        # Recognition loss
        output, aux_output = y
        output = torch.nn.functional.log_softmax(output, dim=-1)

        loss = ctc_loss(output.cpu(), y_enc, x_reduced_len, y_len_enc)
        
        loss_main_epoch += loss.item()

        if args.train_strategy == "cnn": 
            loss_shortcut_epoch = 0.0
        else:
            aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
            loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc, x_reduced_len, y_len_enc)
            loss_shortcut_epoch += loss_shortcut.item()
            loss += loss_shortcut

        loss.backward()
        optimizer.step()

    loss_main_epoch /= (index_batch + 1)
    loss_shortcut_epoch /= (index_batch + 1)

    losses = {
        "loss_main": loss_main_epoch,
        "loss_shortcut": loss_shortcut_epoch,
    }

    return losses


def train_crnn_reg_one_epoch(training_loader,
                             optimizer,
                             model,
                             device,
                             ctc_loss,
                             conf_reg,
                             clusters,
                             char_list,
                             token_blank,
                             text_read,
                             weight_loss_shortcut=0.1):
    loss_enc_main_epoch = 0
    loss_enc_shortcut_epoch = 0
    loss_reg_epoch = 0

    model.train()

    for index_batch, batch_data in enumerate(training_loader):
        print("index_batch: ", index_batch)
        optimizer.zero_grad()

        x = batch_data["imgs"].to(device)
        x_reduced_len = batch_data["w_reduce"]

        y_enc = batch_data["label_ind"].to(device)
        y_len_enc = batch_data["label_ind_length"]

        y_gt_txt = batch_data["label_str"]  # No need  remove space padding

        y, _, after_blstm = model(x)

        # Recognition loss
        output, aux_output = y

        output = torch.nn.functional.log_softmax(output, dim=-1)

        # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
        output_cer = output.transpose(0, 1)

        top_frames_main_pred = [torch.argmax(lp, dim=1)[:x_reduced_len[j]] for j, lp in enumerate(output_cer)]

        top_main_enc_cer = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(output_cer)]
        # No need  remove space padding
        predictions_text_main_enc = [text_read.ctc_best_path_one(p, char_list, token_blank) if p is not None else "" for
                                     p in top_main_enc_cer]

        cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

        # Get sequence frames labels
        gt_frame_format_ok = []

        for i_item in range(len(top_frames_main_pred)):
            if cers[i_item] == 0:
                gt_frame_format_ok.append(top_frames_main_pred[i_item])
            else:
                gt_frame_format_ok.append(None)

        # Recognition losses
        loss = ctc_loss(output.cpu(), y_enc, x_reduced_len, y_len_enc)
        loss_enc_main_epoch += loss.detach().item()  # loss.item()
        print("loss_enc_main_epoch", loss_enc_main_epoch)

        aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
        loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc, x_reduced_len, y_len_enc)
        loss_enc_shortcut_epoch += loss_shortcut.detach().item()

        loss += loss_shortcut

        # Center Loss
        after_blstm = torch.permute(after_blstm, (1, 0, 2))
        # Apply activation function
        after_blstm = torch.sigmoid(after_blstm)

        dict_feature_per_class_ok = groupe_features_per_class(after_blstm, gt_frame_format_ok, conf_reg["index_class_to_filter"])

        loss_reg_ok = compute_center_loss_k1(dict_feature_per_class_ok, clusters, conf_reg["loss_reg"])

        loss_reg = conf_reg["weight_loss_regularization_ok"] * loss_reg_ok

        # Cas all predictions are classes filtered or cer != 0
        if not isinstance(loss_reg, float) and not isinstance(loss_reg, int):
            loss_reg = loss_reg.to(loss.device)
            loss += loss_reg
            loss_reg_epoch += loss_reg.detach().item()

        loss.backward()

        optimizer.step()

    loss_enc_main_epoch /= (index_batch + 1)
    loss_enc_shortcut_epoch /= (index_batch + 1)
    loss_reg_epoch /= (index_batch + 1)

    losses = {
        "loss_main": loss_enc_main_epoch,
        "loss_shortcut": loss_enc_shortcut_epoch,
        "loss_reg_epoch": loss_reg_epoch,
    }
    print(losses)

    return losses
