"""
 Copyright (c) 2018-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import datetime
from functools import partial

import cv2 as cv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as t

from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np


def get_subset(container, subset_bounds):
    """Returns a subset of the given list with respect to the list of bounds"""
    subset = []
    for bound in subset_bounds:
        subset += container[bound[0]: bound[1]]
    return subset


def get_roc(scores_with_gt, n_threshs=400):
    """Computes a ROC cureve on the LFW dataset"""
    thresholds = np.linspace(0., 4., n_threshs)

    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        fp = 0
        tp = 0
        for score_with_gt in scores_with_gt:
            predict_same = score_with_gt['score'] < threshold
            actual_same = score_with_gt['is_same']

            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1

        fp_rates.append(float(fp) / len(scores_with_gt) * 2)
        tp_rates.append(float(tp) / len(scores_with_gt) * 2)

    return np.array(fp_rates), np.array(tp_rates)


def get_auc(fprs, tprs):
    """Computes AUC under a ROC curve"""
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def save_roc(fp_rates, tp_rates, fname):
    assert fp_rates.shape[0] == tp_rates.shape[0]
    with open(fname + '.txt', 'w') as f:
        for i in range(fp_rates.shape[0]):
            f.write('{} {}\n'.format(fp_rates[i], tp_rates[i]))


@torch.no_grad()
def compute_embeddings_lfw(val_loader, model,
                           pdist=lambda x, y: 1. - F.cosine_similarity(x, y), flipped_embeddings=False):
    """Computes embeddings of all images from the LFW dataset using PyTorch"""
    scores_with_gt = []
    embeddings = []
    ids = []
    batch_size = 0

    for batch_idx, data in enumerate(tqdm(val_loader, 'Computing embeddings')):
        images_1 = data['img1']
        images_2 = data['img2']
        if batch_size == 0:
            batch_size = data['img1'].shape[0]
        is_same = data['is_same']
        #if torch.cuda.is_available() and args.devices[0] != -1:
        #    images_1 = images_1.cuda()
        #    images_2 = images_2.cuda()
        emb_1 = model(images_1)
        emb_2 = model(images_2)
        if flipped_embeddings:
            images_1_flipped = torch.flip(images_1, 3)
            images_2_flipped = torch.flip(images_2, 3)
            emb_1_flipped = model(images_1_flipped)
            emb_2_flipped = model(images_2_flipped)
            emb_1 = (emb_1 + emb_1_flipped)*.5
            emb_2 = (emb_2 + emb_2_flipped)*.5
        scores = pdist(emb_1, emb_2).data.cpu().numpy()

        for i, _ in enumerate(scores):
            scores_with_gt.append({'score': scores[i], 'is_same': is_same[i], 'idx': batch_idx*batch_size + i})

    return scores_with_gt


def compute_optimal_thresh(scores_with_gt):
    """Computes an optimal threshold for pairwise face verification"""
    pos_scores = []
    neg_scores = []
    for score_with_gt in scores_with_gt:
        if score_with_gt['is_same']:
            pos_scores.append(score_with_gt['score'])
        else:
            neg_scores.append(score_with_gt['score'])

    hist_pos, bins = np.histogram(np.array(pos_scores), 60)
    hist_neg, _ = np.histogram(np.array(neg_scores), bins)

    intersection_bins = []

    for i in range(1, len(hist_neg)):
        if hist_pos[i - 1] >= hist_neg[i - 1] and 0.05 < hist_pos[i] <= hist_neg[i]:
            intersection_bins.append(bins[i])

    if not intersection_bins:
        intersection_bins.append(0.5)

    return np.mean(intersection_bins)


def evaluate_lfw(dataset, model, roc_fname='', verbose=True, show_failed=False):
    """Computes the LFW score of given model"""
    if verbose and isinstance(model, torch.nn.Module):
        log.info('Face recognition model config:')
        log.info(model)

    scores_with_gt = compute_embeddings_lfw(dataset, model)
    num_pairs = len(scores_with_gt)

    subsets = []
    for i in range(10):
        lower_bnd = i * num_pairs // 10
        upper_bnd = (i + 1) * num_pairs // 10
        subset_test = [(lower_bnd, upper_bnd)]
        subset_train = [(0, lower_bnd), (upper_bnd, num_pairs)]
        subsets.append({'test': subset_test, 'train': subset_train})

    same_scores = []
    diff_scores = []
    val_scores = []
    threshs = []
    mean_fpr = np.zeros(400)
    mean_tpr = np.zeros(400)
    failed_pairs = []

    for subset in tqdm(subsets, 'LFW evaluation', disable=not verbose):
        train_list = get_subset(scores_with_gt, subset['train'])
        optimal_thresh = compute_optimal_thresh(train_list)
        threshs.append(optimal_thresh)

        test_list = get_subset(scores_with_gt, subset['test'])
        same_correct = 0
        diff_correct = 0
        pos_pairs_num = neg_pairs_num = len(test_list) // 2

        for score_with_gt in test_list:
            if score_with_gt['score'] < optimal_thresh and score_with_gt['is_same']:
                same_correct += 1
            elif score_with_gt['score'] >= optimal_thresh and not score_with_gt['is_same']:
                diff_correct += 1

            if score_with_gt['score'] >= optimal_thresh and score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])
            if score_with_gt['score'] < optimal_thresh and not score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])

        same_scores.append(float(same_correct) / pos_pairs_num)
        diff_scores.append(float(diff_correct) / neg_pairs_num)
        val_scores.append(0.5*(same_scores[-1] + diff_scores[-1]))

        fprs, tprs = get_roc(test_list, mean_fpr.shape[0])
        mean_fpr = mean_fpr + fprs
        mean_tpr = mean_tpr + tprs

    mean_fpr /= 10
    mean_tpr /= 10

    if roc_fname:
        save_roc(mean_tpr, mean_fpr, roc_fname)

    same_acc = np.mean(same_scores)
    diff_acc = np.mean(diff_scores)
    overall_acc = np.mean(val_scores)
    auc = get_auc(mean_fpr, mean_tpr)

    if show_failed:
        log.info('Number of misclassified pairs: {}'.format(len(failed_pairs)))
        for pair in failed_pairs:
            dataset.show_item(pair)

    avg_optimal_thresh = np.mean(threshs)
    if verbose:
        log.info('Accuracy/Val_same_accuracy mean: {0:.4f}'.format(same_acc))
        log.info('Accuracy/Val_diff_accuracy mean: {0:.4f}'.format(diff_acc))
        log.info('Accuracy/Val_accuracy mean: {0:.4f}'.format(overall_acc))
        log.info('Accuracy/Val_accuracy std dev: {0:.4f}'.format(np.std(val_scores)))
        log.info('AUC: {0:.4f}'.format(auc))
        log.info('Estimated threshold: {0:.4f}'.format(avg_optimal_thresh))
    return same_acc, diff_acc, overall_acc, auc, avg_optimal_thresh
