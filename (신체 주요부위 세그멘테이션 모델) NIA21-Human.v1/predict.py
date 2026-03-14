# ------------------------------------------------------------------------------------------------------------------- #
import argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils import data

from libs.polygon2mask import mask2polygons, polygon2json
from libs.polygon2mask import class_color, class_color_code
from libs.make_png import class_code
torch.multiprocessing.set_start_method("spawn", force=True)
import torch.backends.cudnn as cudnn

from libs.encoding import DataParallelModel, DataParallelCriterion
from libs.criterion import CriterionAll
from libs.CE2P import Res_Deeplab
from libs.datasets import NIA2DataSet
from libs.transforms import transform_parsing
from libs.make_png import prepare_prediction_data

import cv2

import logging
from loguru import logger
from pathlib import Path
import os
from datetime import datetime
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
def init_logger(log_path):
    logger.remove()
    logger.add(logging.StreamHandler(), colorize=True, 
        format='<green>[{time:MM-DD HH:mm:ss}]</green><cyan>[{function:17s}({line:3d})] </cyan><level>{message}</level>')
    logger.add(log_path.joinpath('CE2P_pred_{time:YYYYMMDD}.log'), 
        format='[{time:YYYY-MM-DD HH:mm:ss}][{name:9s}][{function:20s}({line:3d})][{level:6s}] {message}')
# ------------------------------------------------------------------------------------------------------------------- #
def init_model(param):
    logger.info(' >> Init model <<')
    no_classes = param.num_classes 
    ignore_value = param.ignore_label 
    logger.info(f'  - num of classes = {no_classes}')
    logger.info(f'  - ignore code    = {ignore_value}')
    deeplab = Res_Deeplab(num_classes=no_classes)
    logger.info(' >> Load weight <<')
    new_params = deeplab.state_dict().copy()
    nia2_restore_from = Path(param.restore_from)
    if nia2_restore_from is not None and nia2_restore_from.is_file():
        logger.info(f'  - Pretrained_restore_from <{nia2_restore_from}>')
        nia2_saved_state_dict = torch.load(nia2_restore_from)
        cnt_from_nia2 = 0
        for i in nia2_saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[1:])] = nia2_saved_state_dict[i]
                cnt_from_nia2 += 1
        logger.info(f'  - Restore weight of {cnt_from_nia2} from <{nia2_restore_from.name}>')
    else:
        logger.error(f'Weigths file should be defined for prediction.')
        raise Exception('No Weight define Error')
    deeplab.load_state_dict(new_params)
    logger.info(' >> Prepare prediction ')
    model = DataParallelModel(deeplab)
    model.cuda()
    criterion = None
    if 'pred' not in param.dataset.lower():
        criterion = CriterionAll(ignore_value)
        criterion = DataParallelCriterion(criterion)
        criterion.cuda()
    return model, criterion, no_classes, ignore_value
# ------------------------------------------------------------------------------------------------------------------- #
def set_paralle_gpus(param):
    gpus_asked = [int(i) for i in param.gpu.replace(' ','').split(',')]
    gpus_asked.sort()

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            gpus_available = [i for i in range(n_gpu)]
            gpus_applied = []
            for g_id in gpus_asked:
                if g_id in gpus_available:
                    gpus_applied.append(g_id)
            gpus_applied.sort()
            if len(gpus_applied) > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus_applied])
                if gpus_asked != gpus_applied:
                    logger.info(f'Applied GPUs are <{gpus_applied}> while asked GPUs are <{gpus_asked}>')
                return gpus_applied
            else:
                logger.warning(f'Asked GPUs as <{gpus_asked}> seems not available.')
                return gpus_applied
    else:
        logger.warning('GPU seems not to be available. It might be stop.')
        return []
# ------------------------------------------------------------------------------------------------------------------- #
def init_data_loader(param, gpus):
    pred_img_path = './temp/pred'
    input_size = [int(size) for size in param.input_size.replace(' ', '').split(',')]
    batch_size = param.batch_size 
    pred_set_fpath = Path(param.datalist_dir).joinpath(param.pred_set)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    pred_dataset, pred_loader, num_pred_samples = None, None, 0
    logger.info(' >> Prepare dataset for prediction <<')
    pred_dataset = NIA2DataSet(logger, pred_img_path, pred_set_fpath, crop_size=input_size, 
                                transform=transform, bArgument=False)
    pred_loader = data.DataLoader(pred_dataset,
                                batch_size=batch_size * len(gpus), 
                                shuffle=True, num_workers=2,
                                pin_memory=True)
    if pred_dataset is not None:
        num_pred_samples = len(pred_dataset)
    logger.info(f'  - Prediction Dataset is loaded as {num_pred_samples}')
    return pred_loader, input_size, pred_img_path, num_pred_samples
# ------------------------------------------------------------------------------------------------------------------- #
def do_data_preparation(param):
    pred_data_path = Path(param.data_dir)
    if pred_data_path.is_dir():
        cnv_list, cnv_images_dic = prepare_prediction_data(pred_data_path)
        with Path(param.datalist_dir).joinpath(param.pred_set).open(mode='w') as fp:
            for data_name in cnv_list:
                fp.write(f'{data_name}\n')
        return cnv_images_dic
    logger.warning(f'Test set file as <{param.pred_set}> at <{param.data_dir}> seems not exist')
    return None
# ------------------------------------------------------------------------------------------------------------------- #
def predict(logger, model, predloader, input_size, num_samples, gpus):
    model.eval()
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)
    im_names = []
    
    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    str_len = len(str(len(predloader)))
    with torch.no_grad():
        for index, batch in enumerate(predloader):
            image, _, _, meta = batch
            num_images = image.size(0)
            if (index+1) % 10 == 0:
                logger.info(f'  - prediction processed as {index*num_images:>{str_len}d}')

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]
            im_names = im_names + meta['name']

            outputs = model(image.cuda())

            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                    idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                idx += num_images
        logger.info(f'  - prediction processed as {index*num_images:>{str_len}d}')
    parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, scales, centers, im_names
# ------------------------------------------------------------------------------------------------------------------- #
def get_color_image(labels, num_classes):
    available_idx = labels < num_classes
    lbl_out = labels * available_idx
    lbl_clr = np.zeros((labels.shape[0],labels.shape[1],3), dtype=np.uint8)
    for code, clr_hex in class_color_code.items():
        rgb = [int(clr_hex['color'][i:i+2], 16) for i in range(1,6,2)]
        for c_idx in [0,1,2]:
            c_part = lbl_clr[:,:,c_idx]
            c_part[lbl_out == code] = rgb[c_idx]
    return lbl_clr
# ------------------------------------------------------------------------------------------------------------------- #
def get_org_image_size(param, im_fpath):
    if Path(im_fpath).is_file():
        im_org = cv2.imread(im_fpath)
        return im_org.shape[1], im_org.shape[0]
    else:
        return 0, 0
# ------------------------------------------------------------------------------------------------------------------- #
def save_predicted_result(param, label_preds, scales, centers, im_names, cov_images_dict):
    save_path = Path(param.output_path)
    save_path.mkdir(exist_ok=True)
    n_total = len(im_names)
    print('> Save result : ', end='', flush=True)
    str_prog = ''
    for idx, name in enumerate(im_names):
        lbl_pred, s, c = label_preds[idx,:,:], scales[idx,:], centers[idx,:]
        w, h = get_org_image_size(param, cov_images_dict.get(name).split(':')[1])
        lbl = transform_parsing(lbl_pred, c, s, w, h, lbl_pred.shape, 255)
        lbl_img_fpath = save_path.joinpath(f'{name}.png')
        lbl_json_fpath = lbl_img_fpath.with_suffix('.json')
        cv2.imwrite(lbl_img_fpath.as_posix(), get_color_image(lbl, param.num_classes))
        lbl_json_fpath.write_text(polygon2json(mask2polygons(lbl)), encoding='utf-8')
        if (idx+1)%int(n_total*0.1) == 0:
            str_prog = f'{int(0.5+100*idx/n_total):3d}%'
            print(f'{str_prog} ', end='', flush=True)
    print(' < Finish to save.' if str_prog == '100%' else '100% < Finish to save.')
# ------------------------------------------------------------------------------------------------------------------- #
def predict_main(param):
    logger.info( ' >> Data preprocessing for prediction Start <<')
    cov_images_dict = do_data_preparation(param)
    logger.info( ' >> Prediction Start <<')
    logger.info(' >> Init cudnn <<')
    cudnn.enabled, cudnn.benchmark = True, True
    torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = False, True
    model, _, no_classes, ignore_value = init_model(param)
    gpus = set_paralle_gpus(param)
    pred_loader, input_size, data_path, num_pred_samples = init_data_loader(param, gpus)
    logger.info(f' >> Inferrence start <<')
    label_preds, scales, centers, im_names = predict(logger, model, pred_loader, 
                                                    input_size, num_pred_samples, len(gpus))
    logger.info(f' >> Save results of prediction start <<')
    save_predicted_result(param, label_preds, scales, centers, im_names, cov_images_dict)
    logger.info(f' >> Prediction finished <<')
# ------------------------------------------------------------------------------------------------------------------- #
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # default value for argument
    GPUIDs='0,1'
    BATCH_SIZE = 8
    DATA_DIR = './input_data'
    DATALIST_DIR = './input_catalog'
    IGNORE_LABEL = 255
    INPUT_SIZE = '735,490'
    NUM_CLASSES = 15 #background should be included
    RESTORE_FROM = './weights/NIA2_epoch_139.pth'
    OUT_PATH = './output/pred'
    DATASET = 'pred'
    PRED_SET = f'pred_{datetime.now().strftime("%Y%m%d%H%M%S")}_id.txt'

    parser = argparse.ArgumentParser(description="NIA21-Human Prediction")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--datalist-dir", type=str, default=DATALIST_DIR,
                        help="Path to the directory containing the dataset list.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--output-path", type=str, default=OUT_PATH,
                        help="Prediction result file folder")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Prediction dataset")
    parser.add_argument("--gpu", type=str, default=GPUIDs,
                        help="choose gpu device.")
    parser.add_argument("--pred-set", type=str, default=PRED_SET,
                        help="Predition data list file name")
    return parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    param = get_arguments()
    Path('./temp').mkdir(exist_ok=True)
    Path(param.output_path).mkdir(exist_ok=True)

    log_path = Path(param.output_path) 
    init_logger(log_path)
    try:
        predict_main(param)
    except Exception as ex:
        logger.exception(ex)
# ------------------------------------------------------------------------------------------------------------------- #
