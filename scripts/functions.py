import numpy as np

import os
import pandas as pd
from collections import defaultdict
import glob


def iou_box(boxes, boxes_pred, scores, thresholds):
    """
    Mean precision at different intersections over IoU threshold.
    :param boxes: Mx4 numpy array of the bounding box. (x1, y1, w, h)
    :param boxes_pred: Nx4 numpy array of the predicted box. (x1, y1, w, h)
    :param scores: length N numpy array of bounding box scores,
    :param thresholds: IoU thresholds.
    :return: mean precision of the image.
    """

    def iou(b1, b2):
        # getting the boxes
        x11, y11, w1, h1 = b1
        x12, y12, w2, h2 = b2

        # getting the other end of the boxes
        x21 = x11 + w1
        y21 = y11 + h1
        x22 = x12 + w2
        y22 = y12 + h2

        # getting the area of the boxes
        a1 = w1 * h1
        a2 = w2 * h2

        assert a1 > 0 and a2 > 0

        xi1 = x11 if x11 > x12 else x12
        xi2 = x21 if x21 > x22 else x22
        yi1 = y11 if y11 > y12 else y12
        yi2 = y21 if y21 > y22 else y22

        if xi1 >= xi2 or yi1 > yi2:
            return 0
        else:
            intersect = (xi2 - xi1) * (yi2 - yi1)
            union = a1 + a2 - intersect
            return intersect / union

    if len(boxes) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes.shape[1] == 4 and boxes_pred.shape[1] == 4, "Boxes should have a shape[1]=4"

    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "Length of scoes and boxes_pred are different."
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_t = 0

    for th in thresholds:
        m_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes):
            matched = False
            for j, bp in enumerate(boxes_pred):
                m = iou(bt, bp)
                if m >= th and not matched and j not in m_bt:
                    matched = True
                    tp += 1
                    m_bt.add(j)
            if not matched:
                fn += 1

        fp = len(boxes_pred) - len(m_bt)
        ma = tp / (tp + fn + fp)
        map_t += ma

    return map_t / len(thresholds)


def iou(t, pred, p_id=None):
    """
    IoU for the entire validation set.
    :param t: dict where key= patient id and value = list of bounding boxes
    :param pred: same as t
    :param p_id: certain patient id
    :return: iou, tp, fp, tn, fn
    """
    ious = []
    tp, fp, tn, fn, = 0, 0, 0, 0

    if p_id != None:
        pass
    else:
        for pid in pred.keys():
            t_len = len(t[pid])
            p_len = len(pred[pid])
            if t_len == 0 and p_len == 0:
                tn += 1
            elif t_len == 0 and p_len > 0:
                fp += 1
                ious.append(0)
            elif t_len > 0 and p_len == 0:
                fn += 1
                ious.append(0)
            elif t_len > 0 and p_len > 0:
                tp += 1

                bt = t[pid]
                bp = [[b[1], b[2], b[3], b[4]] for b in pred[pid]]
                scores = [b[0] for b in pred[pid]]
                ious.append(iou_box(np.array(bt), np.array(bp), scores))
        return np.array(ious).mean(), tp, fp, tn, fn


def box_locations():
    locations = defaultdict(list)
    labels = pd.read_csv(os.path.join('../data/stage_2_train_labels.csv'))

    for a, row in labels.iterrows():
        if row.Target == 1:
            locations[row.patientId].append([int(row.x), int(row.y), int(row.width), int(row.height)])

    return locations


def create_output(pred, output='output.csv'):
    file = open(output, 'w')
    file.write("patientId, PredictionString\n")
    for pid, b in pred.items():
        s = pid + ","
        if len(b) != 0:
            for box in b:
                s += " " + str(round(box[0], 2)) + " "
                s2 = "{} {} {} {}".format(box[1], box[2], box[3], box[4])
                s += s2
            s += "\n"
        file.write(s)
    file.close()


def get_image_fps(dir):
    image_fps = glob.glob(dir + '/' + '*.dcm')
    return list(set(image_fps))