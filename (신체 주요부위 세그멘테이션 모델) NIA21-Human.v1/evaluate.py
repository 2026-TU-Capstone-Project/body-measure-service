# ------------------------------------------------------------------------------------------------------------------- #
import argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils import data
torch.multiprocessing.set_start_method("spawn", force=True)
import torch.backends.cudnn as cudnn

from libs.encoding import DataParallelModel, DataParallelCriterion
from libs.criterion import CriterionAll
from libs.CE2P import Res_Deeplab
from libs.datasets import NIA2DataSet
from libs.miou import compute_mean_ioU
from libs.make_png import prepare_validation_data

import logging
from loguru import logger
from pathlib import Path
import os
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
def init_logger(log_path):
    logger.remove()
    logger.add(logging.StreamHandler(), colorize=True, 
        format='<green>[{time:YYYY-MM-DD HH:mm:ss Z}]</green><cyan>[{function:17s}({line:3d})] </cyan><level>{message}</level>')
    logger.add(log_path.joinpath('CE2P_eval_{time:YYYYMMDD}.log'), 
        format='[{time:YYYY-MM-DD HH:mm:ss Z}][{name:9s}][{function:20s}({line:3d})][{level:6s}] {message}')
# ------------------------------------------------------------------------------------------------------------------- #
def valid(logger, model, valloader, input_size, num_samples, gpus, criterion):
    model.eval()
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)
    im_names = []
    
    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    str_len = len(str(len(valloader)))
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, labels, edges, meta = batch
            num_images = image.size(0)
            if (index+1) % 10 == 0:
                logger.info(f'  - validation processed as {index*num_images:>{str_len}d}')

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]
            im_names = im_names + meta['name']

            outputs = model(image.cuda())

            loss = None
            if len(labels) > 0 and len(edges) > 0:
                labels, edges = labels.long().cuda(non_blocking=True), edges.long().cuda(non_blocking=True)
                loss = None if criterion is None else criterion(outputs, [labels, edges])

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
        logger.info(f'  - validation processed as {index*num_images:>{str_len}d}')
    parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers, im_names, loss
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
        logger.error(f'Weigths file should be defined for validation.')
        raise Exception('No Weight define Error')
    deeplab.load_state_dict(new_params)
    logger.info(' >> Prepare validation ')
    model = DataParallelModel(deeplab)
    model.cuda()
    criterion = CriterionAll(ignore_value)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    return model, criterion, no_classes, ignore_value
# ------------------------------------------------------------------------------------------------------------------- #
def set_paralle_gpus():
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            gpus = [i for i in range(n_gpu)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
        return gpus
    else:
        logger.warning('GPU seems not to be available. It might be stop.')
        return []
# ------------------------------------------------------------------------------------------------------------------- #
def init_data_loader(param, gpus):
    data_path = Path(param.data_dir)
    input_size = [int(size) for size in param.input_size.replace(' ', '').split(',')]
    batch_size = param.batch_size 
    test_set_fpath = Path(param.datalist_dir).joinpath(param.test_set)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset, test_loader, num_test_samples = None, None, 0
    logger.info(' >> Prepare dataset for validation <<')
    test_dataset = NIA2DataSet(logger, data_path, test_set_fpath, crop_size=input_size, 
                                transform=transform, bArgument=False)
    test_loader = data.DataLoader(test_dataset,
                                batch_size=batch_size * len(gpus), 
                                shuffle=True, num_workers=2,
                                pin_memory=True)
    if test_dataset is not None:
        num_test_samples = len(test_dataset)
    logger.info(f'  - Validation Dataset is loaded as {num_test_samples}')
    return test_loader, input_size, data_path, num_test_samples
# ------------------------------------------------------------------------------------------------------------------- #
def do_data_preparation(param):
    test_set_fpath = Path(param.datalist_dir).joinpath(param.test_set)
    if test_set_fpath.is_file():
        prepare_validation_data(test_set_fpath)
        return True
    logger.warning(f'Test set file as <{param.test_set}> at <{param.datalist_dir}> seems not exist')
    return False
# ------------------------------------------------------------------------------------------------------------------- #
def evaluate_main(param):
    logger.info( ' >> Data preprocessing for validation Start <<')
    do_data_preparation(param)
    logger.info( ' >> Validation Start <<')
    logger.info(' >> Init cudnn <<')
    cudnn.enabled, cudnn.benchmark = True, True
    torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = False, True
    model, criterion, no_classes, ignore_value = init_model(param)
    gpus = set_paralle_gpus()
    test_loader, input_size, data_path, num_test_samples = init_data_loader(param, gpus)
    logger.info(f' >> Validation start <<')
    test_preds, scales, centers, im_names, _ = valid(logger, model, test_loader, 
                                                    input_size, num_test_samples, len(gpus), None)
    logger.info(f'  > Compute evaluation matrics <')
    mIoU = compute_mean_ioU(logger, im_names, test_preds, scales, centers, no_classes, data_path, input_size, ignore_value,
                            bSave=True, save_fpath=Path(param.output_path).joinpath(param.output_file))
    for k, v in mIoU.items():
        logger.info(f'  - {v:6.2f} : {k}')
    logger.info(f' >> Validation finished <<')
# ------------------------------------------------------------------------------------------------------------------- #
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # default value for argument
    BATCH_SIZE = 8
    DATA_DIR = './temp/data'
    IGNORE_LABEL = 255
    INPUT_SIZE = '735,490'
    NUM_CLASSES = 15
    RESTORE_FROM = './weights/NIA2_best_weight.pth'
    DATALIST_DIR = './input_catalog'
    TESTSET = 'test_id.txt'
    OUT_FANME = 'miou_details.csv'
    OUT_PATH = './output'

    parser = argparse.ArgumentParser(description="NIA21-Human Evaluation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--datalist-dir", type=str, default=DATALIST_DIR,
                        help="Path to the directory containing the data ID list file.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--test-set", type=str, default=TESTSET,
                        help="Validation data list file name")
    parser.add_argument("--output-path", type=str, default=OUT_PATH,
                        help="Validation result file folder")
    parser.add_argument("--output-file", type=str, default=OUT_FANME,
                        help="Validation result file name")
    return parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    param = get_arguments()
    Path('./temp').mkdir(exist_ok=True)

    log_path = Path(param.output_path) 
    init_logger(log_path)
    try:
        evaluate_main(param)
    except Exception as ex:
        logger.exception(ex)
# ------------------------------------------------------------------------------------------------------------------- #

