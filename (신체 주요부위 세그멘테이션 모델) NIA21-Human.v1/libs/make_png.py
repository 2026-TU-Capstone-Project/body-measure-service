# ------------------------------------------------------------------------------------------------------------------- #
import cv2
from cv2 import VIDEOWRITER_PROP_QUALITY
from torch import int16
from libs import polygon2mask
from pathlib import Path
from loguru import logger
import logging
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
logger.remove()    
logger.add(logging.StreamHandler(), colorize=True, 
    format='<green>[{time:YYYY-MM-DD HH:mm:ss}]</green><cyan>[{name:9s}][{function:20s}({line:3d})] </cyan><level>{message}</level>')
# ------------------------------------------------------------------------------------------------------------------- #
class_code = {
    "머리":0,
    "몸통":1,
    "위왼팔":2,
    "아래왼팔":3,
    "위오른팔":4,
    "아래오른팔":5,
    "왼손":6,
    "오른손":7,
    "위오른쪽다리":8,
    "아래오른쪽다리":9,
    "위왼쪽다리":10,
    "우왼쪽다리":10,
    "아래왼쪽다리":11,
    "위아랫쪽다리":11,
    "왼발":12,
    "오른발":13,
    "배경":14,
}
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
@logger.catch
def make_png_from_json(conveter_id_dic:dict, reduce_sacle, background, target_path:Path = Path('./temp/data')):
    if background is not None:
        logger.info(f'Background is set as {background}')
    else:
        logger.info('Backgroud is not set')
    target_path.mkdir(exist_ok=True)
    img_path = target_path.joinpath('image')
    img_path.mkdir(exist_ok=True)
    lbl_path = target_path.joinpath('label')
    lbl_path.mkdir(exist_ok=True)
    edge_path = target_path.joinpath('edge')
    edge_path.mkdir(exist_ok=True)

    conveted_id_list = []
    cnt, nskipped, nprocessed = 0, 0, 0
    print('> Start :', end='', flush=True)
    for data_id, fpathes_info in conveter_id_dic.items():
        tkns = fpathes_info.split(':')
        fpath_label = Path(tkns[0])
        fpath_image = Path(tkns[1]) if tkns != 'None' else None
        dn = fpath_label.parent.parent.name
        if fpath_label.is_file() and fpath_image is not None and fpath_image.is_file():
            tar_lbl_path = lbl_path.joinpath(dn)
            tar_lbl_path.mkdir(exist_ok=True)
            tar_img_path = img_path.joinpath(dn)
            tar_img_path.mkdir(exist_ok=True)
            tar_edge_path = edge_path.joinpath(dn)
            tar_edge_path.mkdir(exist_ok=True)
            try:
                img_fpath = tar_img_path.joinpath(fpath_label.name.replace('.json','.jpg'))
                png_fpath = tar_lbl_path.joinpath(fpath_label.name.replace('.json', '.png'))
                edge_fpath = tar_edge_path.joinpath(fpath_label.name.replace('.json', '.png'))
                src_img_fpath = fpath_image
                if src_img_fpath is not None and src_img_fpath.is_file():
                    if not png_fpath.is_file():
                        pgns = polygon2mask.json2polygon(
                                polygon2mask.load_pgn_json(fpath_label), 
                                reduce_sacle, reduce_sacle
                            )
                        if pgns is not None:
                            im_org = cv2.imread(str(src_img_fpath), cv2.IMREAD_COLOR)
                            im_reduced = cv2.resize(im_org, dsize=(0,0), 
                                                    fx=reduce_sacle, fy=reduce_sacle, interpolation=cv2.INTER_AREA)
                            size = im_reduced.shape[:2]
                            bkg_value = background if len(pgns) >= 10 else 255
                            mask, edge = polygon2mask.polygons2mask(logger, size, pgns, class_code, bkg_value)
                            cv2.imwrite(str(png_fpath), mask)
                            cv2.imwrite(str(edge_fpath), edge)
                            cv2.imwrite(str(img_fpath), im_reduced)
                            nprocessed += 1
                        else:
                            nskipped += 1
                    else:
                        nskipped += 1
                    conveted_id_list.append(fpath_label.name[:-5])
                else:
                    pass
            except Exception as ex:
                logger.exception(f'Exp for {str(fpath_label.name)} : {ex}')
        cnt += 1
        if len(conveter_id_dic)>10 and cnt%int(len(conveter_id_dic)*0.1) == 0:
            print(f' {int(0.5+100*cnt/float(len(conveter_id_dic))):3d}%', end='', flush=True)
    print(' > End')
    return conveted_id_list
# ------------------------------------------------------------------------------------------------------------------- #
def prepare_validation_data(val_id_fpath): 
    reduce_scale_valid = 1.0
    val_id_list = val_id_fpath.open(mode='r').readlines()
    val_id_dic = {}
    for data_id in val_id_list:
        fn = data_id.strip()
        dn = fn.split('_')[2]
        img_fpath = Path('./input_data').joinpath(dn, 'image', fn+'.jpg')
        lbl_fpath = Path('./input_data').joinpath(dn, 'label', fn+'.json')
        if img_fpath.is_file() and lbl_fpath.is_file():
            val_id_dic[fn] = f'{lbl_fpath.as_posix()}:{img_fpath.as_posix()}'
    val_cnv = make_png_from_json(val_id_dic, reduce_scale_valid, class_code['배경'])
    print(f'> Valid set : input={len(val_id_list)} output={len(val_cnv)}')
# ------------------------------------------------------------------------------------------------------------------- #
def make_img_for_prediction(conveter_id_dic:dict, reduce_sacle, background, target_path:Path = Path('./temp/pred')):
    if background is not None:
        logger.info(f'Background is set as {background}')
    else:
        logger.info('Backgroud is not set')
    target_path.mkdir(exist_ok=True)
    tar_img_path = target_path.joinpath('image')
    tar_img_path.mkdir(exist_ok=True)

    conveted_id_list = []
    cnt, nskipped, nprocessed = 0, 0, 0
    print('> Start :', end='', flush=True)
    for data_id, fpathes_info in conveter_id_dic.items():
        tkns = fpathes_info.split(':')
        src_fpath_image = Path(tkns[1]) if tkns != 'None' else None
        if src_fpath_image.is_file():
            try:
                img_fpath = tar_img_path.joinpath(src_fpath_image.name)
                if src_fpath_image is not None and src_fpath_image.is_file():
                    if not img_fpath.is_file():
                        im_org = cv2.imread(str(src_fpath_image), cv2.IMREAD_COLOR)
                        im_reduced = cv2.resize(im_org, dsize=(0,0), 
                                                fx=reduce_sacle, fy=reduce_sacle, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(img_fpath), im_reduced)
                        nprocessed += 1
                    else:
                        nskipped += 1
                    conveted_id_list.append(src_fpath_image.with_suffix('').name)
                else:
                    pass
            except Exception as ex:
                logger.exception(f'Exp for {str(src_fpath_image.name)} : {ex}')
        cnt += 1
        if len(conveter_id_dic)>10 and cnt%int(len(conveter_id_dic)*0.1) == 0:
            print(f' {int(0.5+100*cnt/float(len(conveter_id_dic))):3d}%', end='', flush=True)
    print(' > End')
    return conveted_id_list
# ------------------------------------------------------------------------------------------------------------------- #
def prepare_prediction_data(pred_img_path): 
    reduce_scale_pred = 1.0
    cand_imgs = [fn for fn in pred_img_path.glob('**/*.jpg')]
    pred_id_dic = {}
    for src_img_fpath in cand_imgs:
        if src_img_fpath.is_file():
            data_name = src_img_fpath.with_suffix('').name
            pred_id_dic[data_name] = f'None:{src_img_fpath.as_posix()}'
    pred_cnv = make_img_for_prediction(pred_id_dic, reduce_scale_pred, class_code['배경'])
    print(f'> Pred set : input={len(cand_imgs)} output={len(pred_cnv)}')
    return pred_cnv, pred_id_dic
# ------------------------------------------------------------------------------------------------------------------- #
