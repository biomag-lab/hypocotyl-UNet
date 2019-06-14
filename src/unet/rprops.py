import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import io, img_as_uint
from skimage.morphology import skeletonize, medial_axis, skeletonize_3d
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from skimage.measure._regionprops import _RegionProperties

from typing import Container
from numbers import Number


class BBox:
    def __init__(self, rprops_bbox):
        min_row, min_col, max_row, max_col = rprops_bbox
        # regionprops bbox representation
        self.min_row = min_row
        self.min_col = min_col
        self.max_row = max_row
        self.max_col = max_col
        self.bbox = rprops_bbox

        # rectangle representation
        self.x, self.y = min_col, min_row
        self.width = max_col - min_col
        self.height = max_row - min_row

        # coordinate representation
        self.P1 = (min_col, min_row)
        self.P2 = (max_col, min_row)
        self.P3 = (min_col, max_row)
        self.P4 = (max_col, max_row)

    def __repr__(self):
        return str(self.bbox)

    def __getitem__(self, item):
        return self.bbox[item]

    def IOU(self, other_bbox):
        # determining the intersection coordinates
        P1_int = (max(self.P1[0], other_bbox.P1[0]),
                  max(self.P1[1], other_bbox.P1[1]))
        P4_int = (min(self.P4[0], other_bbox.P4[0]),
                  min(self.P4[1], other_bbox.P4[1]))

        # check for intersections
        if (P1_int[0] > P4_int[0]) or (P1_int[1] > P4_int[1]):
            return 0

        intersection_area = (P4_int[0] - P1_int[0]) * (P4_int[1] - P1_int[1])
        union_area = self.area() + other_bbox.area() - intersection_area

        return intersection_area / union_area

    def area(self):
        return self.width * self.height


class Hypo:
    def __init__(self, rprops, dpm=False):
        self.length = rprops.area
        if dpm:
            self.length /= dpm
        self.bbox = BBox(rprops.bbox)

    def __repr__(self):
        return "[%d, %s]" % (self.length, self.bbox)

    def IOU(self, other_hypo):
        return self.bbox.IOU(other_hypo.bbox)


class HypoResult:
    def __init__(self, rprops_or_hypos, dpm=False):
        if isinstance(rprops_or_hypos[0], Hypo):
            self.hypo_list = rprops_or_hypos
        elif isinstance(rprops_or_hypos[0], _RegionProperties):
            self.hypo_list = [Hypo(rprops, dpm) for rprops in rprops_or_hypos]
        self.gt_match = None

    def __getitem__(self, item):
        if isinstance(item, Number):
            return self.hypo_list[item]
        if isinstance(item, Container):
            # check the datatype of the list
            if isinstance(item[0], np.bool_):
                item = [idx for idx, val in enumerate(item) if val]
            return HypoResult([self.hypo_list[idx] for idx in item])

    def __len__(self):
        return len(self.hypo_list)

    def mean(self):
        return np.mean([hypo.length for hypo in self.hypo_list])

    def std(self):
        return np.std([hypo.length for hypo in self.hypo_list])

    def score(self, gt_hyporesult, match_threshold=0.5):
        scores = []
        hypo_ious = np.zeros((len(self), len(gt_hyporesult)))
        objectwise_df = pd.DataFrame(columns=['algorithm', 'ground truth'], index=range(len(gt_hyporesult)))

        for hypo_idx, hypo in enumerate(self.hypo_list):
            hypo_ious[hypo_idx] = np.array([hypo.IOU(gt_hypo) for gt_hypo in gt_hyporesult])
            best_match = np.argmax(hypo_ious[hypo_idx])

            # a match is found if the intersection over union metric is
            # larger than the given threshold
            if hypo_ious[hypo_idx][best_match] > match_threshold:
                # calculate the accuracy of the measurement
                gt_hypo = gt_hyporesult[best_match]
                error = abs(hypo.length - gt_hypo.length)
                scores.append(1 - error/gt_hypo.length)

        gt_hypo_ious = hypo_ious.T
        for gt_hypo_idx, gt_hypo in enumerate(gt_hyporesult):
            objectwise_df.loc[gt_hypo_idx, 'ground truth'] = gt_hypo.length
            best_match = np.argmax(gt_hypo_ious[gt_hypo_idx])
            if gt_hypo_ious[gt_hypo_idx][best_match] > match_threshold:
                objectwise_df.loc[gt_hypo_idx, 'algorithm'] = self.hypo_list[best_match].length

        # precision, recall
        self.gt_match = np.apply_along_axis(np.any, 0, hypo_ious > match_threshold)
        self.match = np.apply_along_axis(np.any, 1, hypo_ious > match_threshold)
        # identified_objects = self[self.match]
        true_positives = self.gt_match.sum()
        precision = true_positives/len(self)
        recall = true_positives/len(gt_hyporesult)

        score_dict = {'accuracy': np.mean(scores),
                      'precision': precision,
                      'recall': recall,
                      'gt_mean': gt_hyporesult.mean(),
                      'result_mean': self.mean(),
                      'gt_std': gt_hyporesult.std(),
                      'result_std': self.std()}

        return score_dict, objectwise_df

    def make_df(self):
        result_df = pd.DataFrame(
            [[hypo.length, *hypo.bbox] for hypo in self.hypo_list],
            columns=['length', 'min_row', 'min_col', 'max_row', 'max_col'],
            index=range(1, len(self)+1)
        )

        return result_df

    def hist(self, gt_hyporesult, export_path):
        lengths = [hypo.length for hypo in self.hypo_list]
        gt_lengths = [hypo.length for hypo in gt_hyporesult]
        histogram_bins = range(0, 500, 10)

        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 15))
            plt.hist(lengths, bins=histogram_bins, color='r', alpha=0.2, label='result')
            plt.hist(gt_lengths, bins=histogram_bins, color='b', alpha=0.2, label='ground truth')
            plt.legend()
            plt.savefig(export_path)
            plt.close('all')

    def filter(self, flt):
        if isinstance(flt, Container):
            min_length, max_length = flt
            self.hypo_list = [h for h in self.hypo_list if min_length <= h.length <= max_length]
        elif isinstance(flt, bool) and flt:
            otsu_thresh = threshold_otsu(np.array([h.length for h in self.hypo_list]))
            self.hypo_list = [h for h in self.hypo_list if otsu_thresh <= h.length]


def bbox_to_rectangle(bbox):
    # bbox format: 'min_row', 'min_col', 'max_row', 'max_col'
    # Rectangle format: bottom left (x, y), width, height
    min_row, min_col, max_row, max_col = bbox
    x, y = min_col, min_row
    width = max_col - min_col
    height = max_row - min_row

    return (x, y), width, height


def get_hypo_rprops(hypo, filter=True, already_skeletonized=False, skeleton_method=skeletonize_3d,
                    return_skeleton=False, dpm=False):
    """
    Args:
        hypo: segmented hypocotyl image
        filter: boolean or list of [min_length, max_length]
    """
    hypo_thresh = (hypo > 0.5)
    if not already_skeletonized:
        hypo_skeleton = label(img_as_uint(skeleton_method(hypo_thresh)))
    else:
        hypo_skeleton = label(img_as_uint(hypo_thresh))

    hypo_rprops = regionprops(hypo_skeleton)
    # filter out small regions
    hypo_result = HypoResult(hypo_rprops, dpm)
    hypo_result.filter(flt=filter)

    if return_skeleton:
        return hypo_result, hypo_skeleton > 0

    return hypo_result


def visualize_regions(hypo_img, hypo_result, export_path=None, bbox_color='r', dpi=50):
    with plt.style.context('seaborn-white'):
        # parameters
        fontsize = 30.0
        linewidth = fontsize / 10.0

        figsize = (hypo_img.shape[0]/dpi, hypo_img.shape[1]/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0,0,1,1]) #plt.subplot(111)
        fig.add_axes(ax)
        ax.imshow(hypo_img)
        for hypo_idx, hypo in enumerate(hypo_result):
            rectangle = patches.Rectangle((hypo.bbox.x, hypo.bbox.y), hypo.bbox.width, hypo.bbox.height,
                                          linewidth=linewidth, edgecolor=bbox_color, facecolor='none')
            ax.add_patch(rectangle)
            ax.text(hypo.bbox.x, hypo.bbox.y - linewidth - 0.8*fontsize, "N.%d." % (hypo_idx+1), fontsize=fontsize, color='k')
            ax.text(hypo.bbox.x, hypo.bbox.y - linewidth, str(hypo.length)[:4], fontsize=fontsize, color=bbox_color)

        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)

        if export_path is None:
            plt.show()
        else:
            plt.savefig(export_path, pad_inches=0, bbox_inches='tight', dpi=dpi)

        plt.close('all')
