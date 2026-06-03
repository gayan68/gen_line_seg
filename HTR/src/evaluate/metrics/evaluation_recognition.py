import editdistance


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = gt.split(" ")
        len_ += len(gt)
    return len_


def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = gt.split(" ")
        pred = pred.split(" ")

        edit += editdistance.eval(gt, pred)
    return edit
