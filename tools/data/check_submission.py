from os.path import exists
import argparse

import numpy as np
from lxml import etree


def eval_market1501(indices, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    num_q = len(q_pids)
    num_g = len(g_pids)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc[:max_rank].cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc[:max_rank].cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc[:max_rank]
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def load_annotation(annot_path):
    tree = etree.parse(annot_path)
    root = tree.getroot()

    assert len(root) == 1
    items = root[0]

    data = dict()
    for item in items:
        image_name = item.attrib['imageName']

        pid = int(item.attrib['vehicleID'])
        cam_id = int(item.attrib['cameraID'][1:])

        image_id = int(image_name.split('.')[0])
        if image_id in data:
            assert ValueError('Image ID {} is duplicated'.format(image_id))

        record = pid, cam_id
        data[image_id] = record

    ordered_image_ids = sorted(data.keys())

    out_pids = np.array([data[key][0] for key in ordered_image_ids])
    out_cam_ids = np.array([data[key][1] for key in ordered_image_ids])

    return out_pids, out_cam_ids


def load_predictions(predictions_path, query_size, gallery_size):
    indices = np.full([query_size, gallery_size], gallery_size + 100, dtype=np.int32)

    data = []
    with open(predictions_path) as input_stream:
        for line in input_stream:
            data.append([int(v) - 1for v in line.replace('\n', '').split(' ')])
            assert min(data[-1]) >= 0
            assert max(data[-1]) < gallery_size
    assert len(data) == query_size

    for query_id, gallery_ids in enumerate(data):
        for i, gallery_id in enumerate(gallery_ids):
            indices[query_id, gallery_id] = i

    return indices.argsort(axis=1)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--submission-file', '-s', type=str, required=True)
    parser.add_argument('--annotation-query', '-q', type=str, required=True)
    parser.add_argument('--annotation-gallery', '-g', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.submission_file)
    assert exists(args.annotation_query)
    assert exists(args.annotation_gallery)

    query_pids, query_cam_ids = load_annotation(args.annotation_query)
    gallery_pids, gallery_cam_ids = load_annotation(args.annotation_gallery)

    indices = load_predictions(args.submission_file, len(query_pids), len(gallery_pids))

    cmc, mAP = eval_market1501(indices, query_pids, gallery_pids, query_cam_ids, gallery_cam_ids)

    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10, 20, 50, 100]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))


if __name__ == '__main__':
    main()
