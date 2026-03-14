# ------------------------------------------------------------------------------------------------------------------- #
from pathlib import Path
from loguru import logger
from datetime import datetime
import logging
import numpy as np
import shutil
from skmultilearn.model_selection import IterativeStratification
from libs.make_png import make_png_from_json
from libs.make_png import class_code
import sys
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
logger.remove()    
logger.add(logging.StreamHandler(), colorize=True, 
    format='<green>[{time:YYYY-MM-DD HH:mm:ss}]</green><cyan>[{name:9s}][{function:20s}({line:3d})] </cyan><level>{message}</level>')
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
def check_valid_data_list(src_img_root_path:Path, src_lbl_root_path:Path, full_label_data_fpath, cnv_id_fname:str):
    assert src_img_root_path.is_dir()
    assert src_lbl_root_path.is_dir()
    
    data_path = Path('./temp/data')
    data_path.mkdir(exist_ok=True)
    img_path = data_path.joinpath('image')
    img_path.mkdir(exist_ok=True)
    lbl_path = data_path.joinpath('label')
    lbl_path.mkdir(exist_ok=True)
    edge_path = data_path.joinpath('edge')
    edge_path.mkdir(exist_ok=True)
    assert img_path.is_dir()
    assert lbl_path.is_dir()
    assert edge_path.is_dir()

    full_label_data_list = full_label_data_fpath.open(mode='r').readlines()
    fp_data_list = open(Path('./input_data').joinpath(cnv_id_fname), 'w')
    convertable_id_dic = {}
    cnt= 0
    print('> Start to make valid data list : ', end='', flush=True)
    for idx, ln in enumerate(full_label_data_list):
        tkns = ln.strip().split(':')
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
                src_img_fpath = fpath_image
                if src_img_fpath is not None and src_img_fpath.is_file():
                    fp_data_list.write(f'{fpath_label.name[:-5]}\n')
                    convertable_id_dic[fpath_label.name[:-5]] = ln.strip()
                    cnt += 1
                else:
                    pass
            except Exception as ex:
                logger.exception(f'Exp for {str(fpath_label.name)} : {ex}')
    print('End <')
    fp_data_list.close()
    return convertable_id_dic
# ------------------------------------------------------------------------------------------------------------------- #
def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = [X[i] for i in train_indices], [y[i] for i in train_indices]
    X_test, y_test = [X[i] for i in test_indices], [y[i] for i in test_indices]
    return X_train, X_test, np.array(y_train), np.array(y_test)
# ------------------------------------------------------------------------------------------------------------------- #
def train_test_split_with_camera_pos(cnv_id_dic:dict, tr_size, vl_size, ts_size):
    #train_test_split using camera pos
    cnv_id_list = list(cnv_id_dic.keys())
    cnv_code_ar = []
    for data_id in cnv_id_list:
        camera_pos = int(data_id.split('_')[-1])
        elem = np.zeros(65, dtype=np.int16)
        elem[camera_pos if 'F' in data_id else camera_pos + 32] = 1
        cnv_code_ar.append(elem)
    tr_id_list, re_id_list, tr_code_ar, re_code_ar = iterative_train_test_split(cnv_id_list, np.array(cnv_code_ar), 
                                                                                train_size=tr_size)
    vl_id_list, ts_id_list, vl_code_ar, ts_code_ar = iterative_train_test_split(re_id_list, re_code_ar, 
                                                                                train_size=(vl_size/(vl_size+ts_size)))
    catalog_path = Path('./input_catalog')
    train_id_dic, val_id_dic, test_id_dic = {}, {}, {}
    with catalog_path.joinpath('train_id.txt').open(mode='w') as fp:
        for v in tr_id_list:
            fp.write(f'{v}\n')
            train_id_dic[v] = cnv_id_dic[v]
    with catalog_path.joinpath('val_id.txt').open(mode='w') as fp:
        for v in vl_id_list:
            fp.write(f'{v}\n')
            val_id_dic[v] = cnv_id_dic[v]
    with catalog_path.joinpath('test_id.txt').open(mode='w') as fp:
        for v in ts_id_list:
            fp.write(f'{v}\n')
            test_id_dic[v] = cnv_id_dic[v]
    return train_id_dic, val_id_dic, test_id_dic
# ------------------------------------------------------------------------------------------------------------------- #
def save_statistics_of_data_list(save_fpath:Path, data_id_dic:dict, title:str):
    id_dic = {}
    for data_id in list(data_id_dic.keys()):
        code = data_id[6]+'_'+data_id.split('_')[-1]
        if code not in id_dic.keys():
            id_dic[code] = 0
        id_dic[code] += 1
    code_ar = list(id_dic.keys())
    code_ar.sort()
    with save_fpath.open(mode='a') as fp:
        fp.write(f'Statistics of {title}\n')
        for key in code_ar:
            fp.write(f'{key} : {id_dic.get(key)}\n')
        fp.write('\n')
# ------------------------------------------------------------------------------------------------------------------- #
def make_train_data(tar_path, total_count, train_id_dic, val_id_dic):
    reduce_scale_train, reduce_scale_valid = 0.25, 1.0
    train_cnv = make_png_from_json(train_id_dic, reduce_scale_train, class_code['배경'], tar_path)
    print(f'> Train set : input={len(train_id_dic)} output={len(train_cnv)} ({100*len(train_id_dic)/total_count:.1f}%')
    val_cnv = make_png_from_json(val_id_dic, reduce_scale_valid, class_code['배경'], tar_path)
    print(f'> Valid set : input={len(val_id_dic)} output={len(val_cnv)} ({100*len(val_id_dic)/total_count:.1f}%')
# ------------------------------------------------------------------------------------------------------------------- #
def _copy_to(cur_fpath, new_fpath):
    if not new_fpath.is_file():
        if not new_fpath.parent.is_dir():
            parent_dirs = [p for p in new_fpath.parents]
            parent_dirs.reverse()
            for p in parent_dirs:
                if not p.is_dir():
                    p.mkdir(exist_ok=True)
        shutil.copy(cur_fpath.as_posix(), new_fpath.as_posix())
# ------------------------------------------------------------------------------------------------------------------- #
def copy_validation_data(test_id_dic):
    data_path = Path('./input_data')
    for data_id, fpathes_info in test_id_dic.items(): 
        tkns = fpathes_info.split(':')
        fpath_label = Path(tkns[0])
        fpath_image = Path(tkns[1])
        dn = data_id.split('_')[2]
        new_fpath_image = data_path.joinpath(dn, 'image', fpath_image.name)
        new_fpath_label = data_path.joinpath(dn, 'label', fpath_label.name)        
        _copy_to(fpath_image, new_fpath_image)
        _copy_to(fpath_label, new_fpath_label)
    print(f'> {len(test_id_dic)} pairs of Test data have been copied to <./input_data>')
# ------------------------------------------------------------------------------------------------------------------- #
def make_dataset_list(src_img_root_path:Path, src_lbl_root_path:Path, full_label_data_fpath, cnv_id_fname:str, 
                      tr_size, vl_size, ts_size):
    cnv_id_dic = check_valid_data_list(src_img_root_path, src_lbl_root_path, full_label_data_fpath, cnv_id_fname)
    train_id_dic, val_id_dic, test_id_dic = train_test_split_with_camera_pos(cnv_id_dic, tr_size, vl_size, ts_size)
    save_fpath = Path(f'./input_catalog/dataset_status_{datetime.now().strftime("%Y%m%d")}.txt')
    save_fpath.write_text('')
    for (title, id_list) in [('Train dataset', train_id_dic), ('Valid dataset', val_id_dic), ('Test dataset', test_id_dic)]:
        save_statistics_of_data_list(save_fpath, id_list, title)
    return train_id_dic, val_id_dic, test_id_dic
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('>>> ----------------------------------------------------------------------------- <<<')
    print('>>> -------                        Data preparaion                        ------- <<<')
    print('>>> ----------------------------------------------------------------------------- <<<')
    print('  > WARNING : This routine only should be runned when Original data is updated.   <')
    print('  > WARNING : This would chanage all data list for train and validation.          <')
    print('>>> ----------------------------------------------------------------------------- <<<')
    answer = input('  >> Do you want to continue (Y/n)? : ')
    if answer.lower() == 'y':
        ans_2nd = input('  >> If you really want to continue, type "YES" : ')
        if ans_2nd == 'YES':
            print('>>> ----------------------------------------------------------------------------- <<<')
            print('  >     Start to data preparation including train and validation dataset split    <')
            print('>>> ----------------------------------------------------------------------------- <<<')
        else:
            sys.exit(0)
    else:
        sys.exit(0)

    src_img_root_path = Path('./input_data')
    src_lbl_root_path = Path('./input_data')
    
    tar_path = Path('./temp/trainval')
    full_label_data_fpath = Path('./input_data/almost_full_label_data.txt')
    if full_label_data_fpath.is_file():
        try:
            tr_s, vl_s, ts_s = 0.8, 0.1, 0.1
            cnv_id_fname = 'converted_data_id.txt'
            train_id_dic, val_id_dic, test_id_dic = make_dataset_list(src_img_root_path, src_lbl_root_path, 
                                                            full_label_data_fpath, cnv_id_fname, tr_s, vl_s, ts_s)
            total_count = len(train_id_dic) + len(val_id_dic) + len(test_id_dic)
            make_train_data(tar_path, total_count, train_id_dic, val_id_dic)
            copy_validation_data(test_id_dic)
        except Exception as ex:
            logger.exception(ex)
    else:
        logger.warning(f'Full label list should be stored in <{full_label_data_fpath.as_posix()}>')
# ------------------------------------------------------------------------------------------------------------------- #
