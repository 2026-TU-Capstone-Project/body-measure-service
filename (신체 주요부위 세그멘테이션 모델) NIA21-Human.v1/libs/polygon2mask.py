# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import cv2
import json
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
class_color_code = {
    0:  {'color':"#1beaac", 'label':"머리",},
    1:  {'color':"#27b73c", 'label':"몸통",},
    2:  {'color':"#56bcec", 'label':"위왼팔",},
    3:  {'color':"#0033cc", 'label':"아래왼팔",},
    4:  {'color':"#6600cc", 'label':"위오른팔",},
    5:  {'color':"#cc66ff", 'label':"아래오른팔",},
    6:  {'color':"#328aa0", 'label':"왼손",},
    7:  {'color':"#f1513c", 'label':"오른손",},
    8:  {'color':"#e9bdff", 'label':"위오른쪽다리",},
    9:  {'color':"#95177f", 'label':"아래오른쪽다리",},
    10: {'color':"#fea518", 'label':"위왼쪽다리",},
    11: {'color':"#f8d016", 'label':"아래왼쪽다리",},
    12: {'color':"#fba39b", 'label':"왼발",},
    13: {'color':"#9999ff", 'label':"오른발",},
    14: {'color':"#ffffff", 'label':"배경",},
}

class_color = {
    '머리':           '#1beaac',
    '몸통':           '#27b73c',
    '위왼팔':         '#56bcec',
    '아래왼팔':       '#0033cc',
    '위오른팔':       '#6600cc',
    '아래오른팔':     '#cc66ff',
    '왼손':           '#328aa0',
    '오른손':         '#f1513c',
    '위오른쪽다리':   '#e9bdff',
    '아래오른쪽다리': '#95177f',
    '위왼쪽다리':     '#fea518',
    '아래왼쪽다리':   '#f8d016',
    '왼발':           '#fba39b',
    '오른발':         '#9999ff',
    '배경':           '#ffffff',
}
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
def polygons2mask(logger, size, polygons, class_code, bkg_value):
    # initialize label mask as background value or ignore value
    mask = np.zeros(size, dtype=np.uint8)
    # background is not None, that means, full labeled case
    if bkg_value is not None:
        if bkg_value != 255 and len(polygons) < 10:
            logger.warning(f'Backgroud value set as <{bkg_value}> while number of polygons are <{len(polygons)}>.')
        mask = mask + bkg_value
    # backgroud is None, that means, partial labeled case
    else:
        mask = mask + 255
    # initialize edge mask as zero
    edge = np.zeros(size, dtype=np.uint8)
    # make ordered polygon dict in order to prevent overlapping mask
    ordered_pgn = {}
    for code in class_code.keys():
        for key, pgn in polygons.items():
            if code in key and key not in ordered_pgn.keys():
                ordered_pgn[key] = pgn
    assert len(ordered_pgn) == len(polygons)
    # make label and edge bitmap
    for key, pgn in ordered_pgn.items():
        code = class_code.get(key)
        if code is None and '_' in key:
            n_key = key.split('_')[0]
            code = class_code.get(n_key)
        edge_color, edge_thickness = 1, 3
        if code is not None:
            pts = [[int(xp), int(yp)] for xp, yp in zip(pgn['x'], pgn['y'])]
            pgn_array = pts if isinstance(pts, np.ndarray) else np.array(pts)
            cv2.fillPoly(mask, [pgn_array], code)
            cv2.polylines(edge, [pgn_array], True, edge_color, edge_thickness)
    return mask, edge
# ------------------------------------------------------------------------------------------------------------------- #
def mask2polygons(mask):
    polygons = {v['label']:[] for v in class_color_code.values()}
    for code, v in class_color_code.items():
        mask4contour = (mask==code).astype(np.uint8)
        if mask4contour.max()>mask4contour.min():
           contour_value, _ = cv2.findContours(mask4contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
           for cnt in contour_value:
               if len(cnt) > 4:
                   polygons[v['label']].append(cnt)
    return polygons
# ------------------------------------------------------------------------------------------------------------------- #
def json2polygon(json_data, scale_x=1.0, scale_y=1.0):
    # check labelingInfo items
    label_list = json_data.get('labelingInfo')
    if label_list is None:
        return None
    # check duplicated label and more than 3 return None
    label_dic = {}
    for label_info in label_list:
        v = label_info['polygon']
        tkns = v['location'].strip().split(' ')
        if len(tkns)%2 == 0 and len(tkns)/2 > 4:
            if v['label'] not in label_dic.keys():
                label_dic[v['label']] = 0
            label_dic[v['label']] += 1
    for k, v in label_dic.items():
        if v > 2 or v == 0:
            return None
    # make polygons
    cords_dic = {}
    for label_info in label_list:
        v = label_info['polygon']
        cords_in_json = v['location']
        tkns = cords_in_json.strip().split(' ')
        x = [float(tkn)*scale_x for i, tkn in enumerate(tkns) if i%2 == 0 ]
        y = [float(tkn)*scale_y for i, tkn in enumerate(tkns) if i%2 == 1 ]
        if len(x) == len(y) and len(x) > 4:
            if v['label'] not in cords_dic.keys():
                cords_dic[v['label']] = {'x':x, 'y':y}
            else:
                for i in range(10):
                    new_label = v['label']+f'_{i+1:2d}'
                    if new_label not in cords_dic.keys():
                        cords_dic[new_label] = {'x':x, 'y':y}
                        break
    return cords_dic
# ------------------------------------------------------------------------------------------------------------------- #
def polygon2json(pgns):
    pgn_json_list = []
    for label, pgn_list in pgns.items():
        for pgn in pgn_list:
            str_pgn_cords = ' '.join([str(i) for i in pgn.flatten()])
            pgn_json_list.append({ 'polygon': {
                    'color': class_color[label],
                    'location': str_pgn_cords,
                    'label': label,
                    'type': 'polygon',
                }})
    return json.dumps({'labelingInfo':pgn_json_list}, indent=4, ensure_ascii=False)
# ------------------------------------------------------------------------------------------------------------------- #
def load_pgn_json(fpath):
    try:
        with fpath.open(mode='r', encoding='utf-8') as fp:
            label_data = json.load(fp)
    except UnicodeDecodeError:
        with fpath.open(mode='r', encoding='cp949') as fp:
            label_data = json.load(fp)
    return label_data
# ------------------------------------------------------------------------------------------------------------------- #
