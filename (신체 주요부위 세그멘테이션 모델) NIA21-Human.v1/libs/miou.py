# ------------------------------------------------------------------------------------------------------------------- #
from unicodedata import name
import numpy as np
import cv2
from collections import OrderedDict
from .transforms import transform_parsing
from .datasets import get_anno_path
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
LABELS = [
        "머리", "몸통", "위왼팔", "아래왼팔", "위오른팔", "아래오른팔", "왼손", "오른손", 
        "위오른쪽다리", "아래오른쪽다리", "위왼쪽다리", "아래왼쪽다리", "왼발", "오른발",
    ]
# ------------------------------------------------------------------------------------------------------------------- #
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
# ------------------------------------------------------------------------------------------------------------------- #
def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
    return confusion_matrix
# ------------------------------------------------------------------------------------------------------------------- #
def compute_mean_ioU(logger, im_names, preds, scales, centers, num_classes, datadir, input_size, ignore_value,
                          bSave=False, save_fpath=None):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i, im_name in enumerate(im_names):
        gt_path = get_anno_path(datadir, im_name) 
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out, s, c = preds[i], scales[i], centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size, ignore_value)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != ignore_value
        gt = gt[ignore_index]
        pred = pred[ignore_index]
        individual_cf = get_confusion_matrix(gt, pred, num_classes)
        if bSave and save_fpath is not None:
            save_individual_results(logger, im_name, individual_cf, save_fpath)
        confusion_matrix += individual_cf
    miou_dic = get_miou_dic(confusion_matrix)
    logger.info(f'  - Pixel accuracy = {miou_dic["Pixel accuracy"]:6.2f}')
    logger.info(f'  - Mean accuracy  = {miou_dic["Mean accuracy"]:6.2f}')
    logger.info(f'  - Mean IoU       = {miou_dic["Mean IoU"]:6.2f}')
    return miou_dic
# ------------------------------------------------------------------------------------------------------------------- #
def get_miou_dic(conf_matrix, start_item=None):
    pos = conf_matrix.sum(1)
    res = conf_matrix.sum(0)
    tp = np.diag(conf_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    
    name_value = []
    if start_item is not None:
        name_value.append(start_item)
    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))
    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IoU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value
# ------------------------------------------------------------------------------------------------------------------- #
def save_individual_results(logger, im_name, indi_cf_matrix, save_fpath):
    miou_dic = get_miou_dic(indi_cf_matrix, ('Data ID', im_name))    
    logger.info(f' {im_name} :  Mean IoU = {miou_dic["Mean IoU"]:6.2f}')
    bNew = not save_fpath.is_file()
    with save_fpath.open(mode='a') as fp:
        if bNew:
            hdr_strs = [k for k in miou_dic.keys()]
            fp.write(','.join(hdr_strs)+'\n')
        out_strs = [v if isinstance(v, str) else f'{v:.2f}' for v in miou_dic.values()]
        fp.write(','.join(out_strs)+'\n')
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
