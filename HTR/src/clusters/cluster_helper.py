import torch
import editdistance
import torch.nn.functional as F

from src.data.text.common_token_txt import CTC_PAD


def groupe_features_per_class(features, gt_seq_frames, index_class_to_filter):
    dict_feature_per_class = {}

    for features_one_item, y_one_item in zip(features, gt_seq_frames):

        if y_one_item is None:
            continue
        # y_one_item: tensor
        for f, y in zip(features_one_item, y_one_item):
            if y.item() in index_class_to_filter:
                continue
            else:
                if y.item() == CTC_PAD:
                    print("Error Pad class is used")
                else:
                    if y.item() in dict_feature_per_class:
                        dict_feature_per_class[y.item()].append(f)
                    else:
                        dict_feature_per_class[y.item()] = [f]

    return dict_feature_per_class


def compute_center_loss_k1(dict_features_per_class, clusters, loss_fct):
    loss_center_all_class = 0

    nb_class = 0
    for index_class, features in dict_features_per_class.items():
        nb_frames_used_class = 0
        loss_one_class = 0
        for one_feature in features:
            index_class_loss = index_class

            loss_reg = loss_fct(one_feature, clusters[index_class_loss])

            loss_one_class += loss_reg
            nb_frames_used_class += 1

        # Norm per class, not all item because classes are unbalanced
        if nb_frames_used_class != 0:
            loss_one_class /= nb_frames_used_class
            nb_class += 1

        loss_center_all_class += loss_one_class

    if nb_class != 0:
        loss_center_all_class /= nb_class

    return loss_center_all_class

#### Simon's version of center computation #####

def compute_center_coordinates(data_loader,
                               model,
                               device,
                               char_list,
                               token_blank,
                               index_class_to_filter,
                               text_read):
    model.eval()

    prototypes_after = torch.zeros([len(char_list), 512]).to(device)

    dict_feature_per_class_after = {}

    # Get prediction features
    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

            y_gt_txt = batch_data["label_str"]

            nb_item_batch = x.shape[0]

            y_pred, _, after_blstm = model(x)

            after_blstm = torch.permute(after_blstm, (1, 0, 2))
            after_blstm = torch.sigmoid(after_blstm)

            # Encoder
            encoder_outputs_main, encoder_outputs_shortcut = y_pred
            encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            encoder_outputs_main = encoder_outputs_main.transpose(0, 1)

            top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(encoder_outputs_main)]
            predictions_text_main_enc = [text_read.ctc_best_path_one(p, char_list, token_blank) if p is not None else ""
                                         for p in
                                         top_main_enc]

            cers_enc = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

            for i in range(nb_item_batch):
                if cers_enc[i] == 0:

                    # Group features by character
                    for f, y in zip(after_blstm[i], top_main_enc[i]):
                        if y in index_class_to_filter:
                            continue
                        if y in dict_feature_per_class_after:
                            dict_feature_per_class_after[y].append(f)
                        else:
                            dict_feature_per_class_after[y] = [f]

    # Compute means
    for key in dict_feature_per_class_after:
        if len(dict_feature_per_class_after[key]) > 0:
            # N, nb features
            features_tensor = torch.stack(dict_feature_per_class_after[key])

            mean_value = torch.mean(features_tensor, 0)
            mean_value = mean_value.detach()

            prototypes_after[key] = mean_value

    return prototypes_after

#### Gayan's version of center computation #####
# def compute_center_coordinates(data_loader,
#                                model,
#                                device,
#                                char_list,
#                                token_blank,
#                                index_class_to_filter,
#                                text_read):
#     model.eval()

#     prototypes_after = torch.zeros([len(char_list), 512]).to(device)

#     batch_dict_feature_per_class_after = {}
#     dict_feature_per_class_after = {}
#     batch_cunter = 1
#     center_cal_batches = 2

#     # Get prediction features
#     with torch.no_grad():
#         for index_batch, batch_data in enumerate(data_loader):
#             x = batch_data["imgs"].to(device)
#             x_reduced_len = batch_data["w_reduce"]

#             y_gt_txt = batch_data["label_str"]

#             nb_item_batch = x.shape[0]

#             y_pred, _, after_blstm = model(x)

#             after_blstm = torch.permute(after_blstm, (1, 0, 2))
#             after_blstm = torch.sigmoid(after_blstm)

#             # Encoder
#             encoder_outputs_main, encoder_outputs_shortcut = y_pred
#             encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

#             # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
#             encoder_outputs_main = encoder_outputs_main.transpose(0, 1)

#             # top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
#             #                 enumerate(encoder_outputs_main)]
#             top_main_enc = torch.argmax(encoder_outputs_main, dim=2)
#             top_main_enc = [top_main_enc[j, :x_reduced_len[j]] for j in range(nb_item_batch)]

#             # keep GPU alive during CPU-heavy work
#             if index_batch % 2 == 0:
#                 _ = torch.randn(1024, 1024, device=device) @ torch.randn(1024, 1024, device=device)


#             predictions_text_main_enc = [text_read.ctc_best_path_one(p, char_list, token_blank) if p is not None else ""
#                                          for p in
#                                          top_main_enc]

#             cers_enc = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

#             for i in range(nb_item_batch):
#                 if cers_enc[i] == 0:

#                     # Group features by character
#                     for f, y in zip(after_blstm[i], top_main_enc[i]):                      
#                         if y in index_class_to_filter:
#                             continue
#                         if y in dict_feature_per_class_after:
#                             dict_feature_per_class_after[y].append(f)
#                         else:
#                             dict_feature_per_class_after[y] = [f]

#             # Compute accumunated batch mean
#             if batch_cunter == center_cal_batches: # Eg: if batch_size =32, and center_cal_batches=16, compute mean (center) every 512 samples. Then reset the dict
#                 for key in dict_feature_per_class_after:
#                     if len(dict_feature_per_class_after[key]) > 0:
#                         # N, nb features
#                         features_tensor = torch.stack(dict_feature_per_class_after[key])

#                         mean_value = torch.mean(features_tensor, 0)
#                         mean_value = mean_value.detach()


#                         if key in batch_dict_feature_per_class_after:
#                             batch_dict_feature_per_class_after[key].append(mean_value)
#                         else:
#                             batch_dict_feature_per_class_after[key] = [mean_value]
                
#                 dict_feature_per_class_after = {}
#                 batch_cunter = 1

#             batch_cunter += 1

#     # Compute final mean form batches
#     for key in batch_dict_feature_per_class_after:
#         if len(batch_dict_feature_per_class_after[key]) > 0:
#             # N, nb features
#             features_tensor = torch.stack(batch_dict_feature_per_class_after[key])

#             mean_value = torch.mean(features_tensor, 0)
#             mean_value = mean_value.detach()

#             prototypes_after[key] = mean_value

#     return prototypes_after



# def compute_center_coordinates(
#     data_loader,
#     model,
#     device,
#     char_list,
#     token_blank,
#     index_class_to_filter,
#     text_read
# ):
#     """
#     GPU-native center computation.
#     No CPU bottlenecks.
#     Keeps GPU active.
#     """

#     model.eval()

#     num_classes = len(char_list)
#     feat_dim = 512  # must match BLSTM output

#     # Global accumulators (GPU)
#     class_sum = torch.zeros(num_classes, feat_dim, device=device)
#     class_count = torch.zeros(num_classes, device=device)

#     index_class_to_filter = torch.tensor(
#         index_class_to_filter, device=device, dtype=torch.long
#     )

#     with torch.no_grad():
#         for batch_data in data_loader:
#             # ----------------------------
#             # Load batch
#             # ----------------------------
#             x = batch_data["imgs"].to(device)               # [B, C, H, W]
#             x_reduced_len = batch_data["w_reduce"]           # list[int]

#             # ----------------------------
#             # Forward pass (GPU active)
#             # ----------------------------
#             y_pred, _, after_blstm = model(x)

#             # after_blstm: [T, B, 512] → [B, T, 512]
#             after_blstm = after_blstm.permute(1, 0, 2)
#             after_blstm = torch.sigmoid(after_blstm)

#             # Encoder outputs
#             encoder_outputs_main, _ = y_pred
#             encoder_outputs_main = F.log_softmax(
#                 encoder_outputs_main, dim=-1
#             )  # [T, B, C]
#             encoder_outputs_main = encoder_outputs_main.transpose(0, 1)  # [B, T, C]

#             # ----------------------------
#             # Argmax decoding (GPU)
#             # ----------------------------
#             top_main_enc = torch.argmax(
#                 encoder_outputs_main, dim=2
#             )  # [B, T]

#             # ----------------------------
#             # Process each sample (short loop, GPU work inside)
#             # ----------------------------
#             for i in range(x.size(0)):
#                 T_i = x_reduced_len[i]

#                 labels = top_main_enc[i, :T_i]           # [T_i]
#                 feats = after_blstm[i, :T_i]             # [T_i, 512]

#                 # Filter ignored classes
#                 valid = ~torch.isin(labels, index_class_to_filter)
#                 labels = labels[valid]
#                 feats = feats[valid]

#                 if labels.numel() == 0:
#                     continue

#                 # ----------------------------
#                 # GPU accumulation
#                 # ----------------------------
#                 class_sum.index_add_(0, labels, feats)
#                 class_count.index_add_(
#                     0,
#                     labels,
#                     torch.ones_like(labels, dtype=torch.float),
#                 )

#     # ----------------------------
#     # Final mean computation
#     # ----------------------------
#     prototypes_after = torch.zeros_like(class_sum)

#     nonzero = class_count > 0
#     prototypes_after[nonzero] = (
#         class_sum[nonzero] / class_count[nonzero].unsqueeze(1)
#     )

#     return prototypes_after
