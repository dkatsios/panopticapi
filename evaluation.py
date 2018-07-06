# #!/usr/bin/env python2
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from utils import get_traceback

OFFSET = 256 * 256 * 256
VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                continue
            n += 1
            pq += iou / (tp + 0.5 * fp + 0.5 * fn)
            sq += iou / tp if tp != 0 else 0
            rq += tp / (tp + 0.5 * fp + 0.5 * fn)

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = pan_gt[:, :, 0] + pan_gt[:, :, 1] * 256 + pan_gt[:, :, 2] * 256 * 256
        print('here')
        print(pred_ann['file_name'].replace('.jpg', '.png'))
        print(pred_ann['file_name'])
        
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = os.path.join(os.path.dirname(gt_json_file), 'segmentations')
    if pred_folder is None:
        pred_folder = os.path.join(os.path.dirname(pred_json_file), 'segmentations')
    categories = {el['id']: el for el in gt_json['categories']}

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            print('no prediction for the image with id: {}'.format(img_id))
            sys.exit(-1)
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name] = pq_stat.pq_average(categories, isthing=isthing)
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
              name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with gorund turth COCO format segmentations. Default: 'segmentations' folder in th same location as gt_json_file.")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. Default: 'segmentations' folder in th same location as pred_json_file")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
