# ------------------------------------------------------------------------------------------------------------------- #
from typing import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary
from torch.utils import data
import torchvision.utils as vutils

from libs.encoding import DataParallelModel, DataParallelCriterion
from libs.criterion import CriterionAll
from libs.CE2P import Res_Deeplab
from libs.datasets import NIA2DataSet
from libs.miou import compute_mean_ioU
from libs.utils import decode_parsing, inv_preprocess
from evaluate import valid

from tensorboardX import SummaryWriter

import os
from pathlib import Path
from loguru import logger
import timeit
import argparse
import logging
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
start = timeit.default_timer()
# ------------------------------------------------------------------------------------------------------------------- #
def init_logger(log_path):
    logger.remove()
    logger.add(logging.StreamHandler(), colorize=True, 
        format='<green>[{time:MM-DD HH:mm:ss}]</green><cyan>[{function:17s}({line:3d})] </cyan><level>{message}</level>')
    logger.add(log_path.joinpath('CE2P_train_{time:YYYYMMDD}.log'), 
        format='[{time:YYYY-MM-DD HH:mm:ss}][{name:9s}][{function:20s}({line:3d})][{level:6s}] {message}')
# ------------------------------------------------------------------------------------------------------------------- #
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
# ------------------------------------------------------------------------------------------------------------------- #
def adjust_learning_rate(optimizer, i_iter, total_iters, base_lr, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(base_lr, i_iter, total_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr
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
    deeplab.load_state_dict(new_params)
    logger.info(' >> Prepare training ')
    model = DataParallelModel(deeplab)
    model.cuda()
    criterion = CriterionAll(ignore_value)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    return model, criterion, no_classes, ignore_value
# ------------------------------------------------------------------------------------------------------------------- #
def set_paralle_gpus(param):
    gpus = [int(g_id) for g_id in param.gpu.split(',')] 
    if param.gpu != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    return gpus
# ------------------------------------------------------------------------------------------------------------------- #
def init_data_loader(param, gpus):
    data_path = Path(param.data_dir) 
    dataset = param.dataset 
    train_set_fpath = Path(param.datalist_dir).joinpath(param.train_set) if 'train' in dataset else None 
    val_set_fpath = Path(param.datalist_dir).joinpath(param.val_set) if 'val' in dataset else None
    input_size = [int(size) for size in param.input_size.replace(' ', '').split(',')]
    batch_size = param.batch_size 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([ transforms.ToTensor(), normalize,])
    tr_dataset, val_dataset, tr_loader, val_loader = None, None, None, None
    if 'train' in dataset:
        logger.info(' >> Prepare dataset for training <<')
        tr_dataset = NIA2DataSet(logger, data_path, train_set_fpath, crop_size=input_size, 
            transform=transform, bArgument=True)
        tr_loader = data.DataLoader(tr_dataset,
                                    batch_size=batch_size * len(gpus), 
                                    shuffle=True, num_workers=2,
                                    pin_memory=True)
        logger.info(f'  - Train Dataset is loaded as {len(tr_dataset)}')
    num_val_samples = 0
    if 'val' in dataset:
        val_dataset = NIA2DataSet(logger, data_path, val_set_fpath, crop_size=input_size, 
            transform=transform, bArgument=False)
        val_loader = data.DataLoader(val_dataset, 
                                    batch_size=batch_size * len(gpus),
                                    shuffle=False, 
                                    pin_memory=True)
        if val_dataset is not None: 
            num_val_samples = len(val_dataset)
        logger.info(f'  - Validation dataset is loaded as {num_val_samples}')
    return tr_loader, val_loader, input_size, data_path, num_val_samples
# ------------------------------------------------------------------------------------------------------------------- #
def log_progress(it_start, total_iters, epochs, epoch, i_iter, loss):
    dur_it = timeit.default_timer() - it_start
    str_len, ep_len = len(str(f'{total_iters:,d}')), len(str(f'{epochs:d}'))
    log_outstr = f'[{epoch+1:{ep_len}d}/{epochs}] iter = {i_iter:{str_len},d}/{total_iters:,d} completed, loss = {loss.data.cpu().numpy():.5f}  ({dur_it:.1f}sec)'
    logger.debug(log_outstr)
# ------------------------------------------------------------------------------------------------------------------- #
def write_interim_results(param, loss_step, image_step, 
                          lr, loss, images, labels, edges, preds, no_classes, writer, i_iter):
    if i_iter % loss_step == 0:
        writer.add_scalar('learning_rate', lr, i_iter)
        writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
    if i_iter % image_step == 0:
        save_num_images = param.save_num_images 
        images_inv = inv_preprocess(images, save_num_images)
        labels_colors = decode_parsing(labels, save_num_images, no_classes, is_pred=False)
        edges_colors = decode_parsing(edges, save_num_images, 2, is_pred=False)
        if isinstance(preds, list):
            preds = preds[0]
        preds_colors = decode_parsing(preds[0][-1], save_num_images, no_classes, is_pred=True)
        pred_edges = decode_parsing(preds[1][-1], save_num_images, 2, is_pred=True)
        img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
        lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
        pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
        edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
        pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)
        writer.add_image('Images/', img, i_iter)
        writer.add_image('Labels/', lab, i_iter)
        writer.add_image('Preds/', pred, i_iter)
        writer.add_image('Edges/', edge, i_iter)
        writer.add_image('PredEdges/', pred_edge, i_iter)
# ------------------------------------------------------------------------------------------------------------------- #
def train_main(param):
    logger.info( ' >> Training Start <<')
    snapshot_path = Path(param.snapshot_dir) 
    writer = SummaryWriter(snapshot_path)
    logger.info(' >> Init cudnn <<')
    cudnn.enabled, cudnn.benchmark = True, True
    torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = False, True
    model, criterion, no_classes, ignore_value = init_model(param)
    gpus = set_paralle_gpus(param)
    tr_loader, val_loader, input_size, data_path, num_val_samples = init_data_loader(param, gpus)
    learning_rate, momentum, weight_decay = param.learning_rate, param.momentum, param.weight_decay
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer.zero_grad()

    start_epoch, epochs, power = param.start_epoch, param.epochs, param.power
    total_iters = epochs * len(tr_loader)
    image_step, loss_step = max(10, int(len(tr_loader)*0.5)), max(10, int(len(tr_loader)*0.1))
    logger.info(f'  - step for loss = {loss_step}, step for image = {image_step}')
    for epoch in range(start_epoch, epochs):
        ep_start = timeit.default_timer()
        model.train()
        for i_iter, batch in enumerate(tr_loader):
            it_start = timeit.default_timer()
            i_iter += len(tr_loader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters, learning_rate, power)
            images, labels, edges, _ = batch
            labels, edges = labels.long().cuda(non_blocking=True), edges.long().cuda(non_blocking=True)
            preds = model(images)
            loss = criterion(preds, [labels, edges])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            write_interim_results(param, loss_step, image_step, 
                          lr, loss, images, labels, edges, preds, no_classes, writer, i_iter)
            log_progress(it_start, total_iters, epochs, epoch, i_iter, loss)
        torch.save(model.state_dict(), snapshot_path.joinpath(f'NIA2_epoch_{epoch:03d}.pth'))
        if num_val_samples > 0: 
            logger.info(f' >> Validation for epoch {epoch+1:d} start <<')
            val_preds, scales, centers, im_names, val_loss = valid(logger, model, val_loader, 
                                                                   input_size, num_val_samples, len(gpus), criterion)
            epoch_loss = OrderedDict([('train loss', loss.data.cpu().numpy()), ('valid loss', val_loss.data.cpu().numpy())])
            writer.add_scalars('LossTrend', epoch_loss, epoch)
            logger.info(f'  - validation loss : {val_loss.data.cpu().numpy():.5f}')
            logger.info(f'  > Compute evaluation matrics <')
            mIoU = compute_mean_ioU(logger, im_names, val_preds, scales, centers, no_classes, data_path, input_size, ignore_value)
            for k, v in mIoU.items():
                logger.info(f'  - {v:6.2f} : {k}')
            writer.add_scalars('mIoU', mIoU, epoch)
            logger.info(f' >> Validation finished <<')
        logger.info(f' >> End of {epoch+1} epoch : duration = {timeit.default_timer() - ep_start:.0f} secs <<')
    # NOTE - log total running time
    end = timeit.default_timer()
    logger.info(f'  >> Training finished through {end - start:.0f} seconds <<')
# ------------------------------------------------------------------------------------------------------------------- #
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # default value for argument
    BATCH_SIZE = 12
    DATA_DIR = './temp/trainval'
    IGNORE_LABEL = 255
    INPUT_SIZE = '735,490'
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    NUM_CLASSES = 15
    POWER = 0.9
    RESTORE_FROM = 'None'
    SAVE_NUM_IMAGES = 12
    SNAPSHOT_DIR = './temp/snapshots/'
    WEIGHT_DECAY = 0.0005
    GPUIDs = '0,1,2,3'
    DATALIST_DIR = './input_catalog'
    TRAINSET = 'train_id.txt'
    VALSET = 'val_id.txt'
    EPOCHS = 100
    START_EPOCH = 0
    DATASET = 'trainval'

    parser = argparse.ArgumentParser(description="NIA21-Human")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--datalist-dir", type=str, default=DATALIST_DIR,
                        help="Path to the directory containing the data ID list file.")
    parser.add_argument("--dataset", type=str, default=DATASET, choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPUIDs,
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="choose the number of recurrence.")
    parser.add_argument("--train-set", type=str, default=TRAINSET,
                        help="Train data list file name")
    parser.add_argument("--val-set", type=str, default=VALSET,
                        help="Validation data list file name")
    return parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    param = get_arguments()
    Path('./temp').mkdir(exist_ok=True)
    Path(param.snapshot_dir).mkdir(exist_ok=True)

    log_path = Path(param.snapshot_dir) 
    init_logger(log_path)
    try:
        train_main(param)
    except Exception as ex:
        logger.exception(ex)
# ------------------------------------------------------------------------------------------------------------------- #
