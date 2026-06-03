import editdistance
import torch

from src.data.text.read_txt_util import Text_Reader
from src.evaluate.metrics.evaluation_recognition import nb_chars_from_list, nb_words_from_list, edit_wer_from_list
from src.evaluate.metrics.metrics_counter import MetricLossCERWER


def evaluate_one_epoch_crnn(data_loader,
                            model,
                            device,
                            char_list,
                            token_blank,
                            ctc_loss_fn,
                            text_read: Text_Reader,
                            compute_wer,
                            use_wer_formula_for_cer,
                            print_all=False):

    metrics_main = MetricLossCERWER("Main")

    model.eval()

    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

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

            ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)

            metrics_main.add_loss(ctc_loss.item(), nb_item_batch)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(output_log)]
            predictions_text = [text_read.ctc_best_path_one(p, char_list, token_blank) for p in top]

            predictions_text = [text_read.remove_space_before_after_one_item(t) for t in predictions_text]  # Remove text padding

            if use_wer_formula_for_cer == 0:
                cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text)]
                metrics_main.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

            else:
                cers = edit_wer_from_list(y_gt_txt, predictions_text)
                metrics_main.add_cer(cers, nb_words_from_list(y_gt_txt))

            if compute_wer == 1:
                metrics_main.add_wer(edit_wer_from_list(y_gt_txt, predictions_text), nb_words_from_list(y_gt_txt))

            # Printing prediction
            if print_all:
                for i in range(nb_item_batch):
                    print("-----Ground truth all:-----")
                    print(y_gt_txt[i])
                    print(y_enc[i])
                    print("-----Predictions:-----")
                    print(predictions_text[i])

            else:
                if index_batch == 0:
                    nb_pred_to_print = min(6, nb_item_batch)
                    for i in range(nb_pred_to_print):
                        print("-----Ground truth all:-----")
                        print(y_gt_txt[i])
                        print("-----Predictions:-----")
                        print(predictions_text[i])

    dict_result = {
        "metrics_main": metrics_main
    }
    return dict_result
