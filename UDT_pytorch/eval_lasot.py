import sys
import json
import os
import glob
from os.path import join as fullfile
import numpy as np

def readall(video_name):
    with open('./dataset/LaSOT/zip/'+video_name+'/'+video_name+'-1'+'/groundtruth.txt','r') as f:
        content = f.readlines()
        contents = []
        for line in content:
            lines = []
            line = line.strip().split(',')
            for i in range(len(line)):
                lines.append(float(line[i]))
            contents.append(lines)
    f.close()
    return contents

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success


def get_result_bb(arch, seq):
    result_path = fullfile(arch, seq + '.txt')
    temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_auc(dataset='vot2018', tracker_reg='S*', start=0, end=1e6):
    list_path = os.path.join('dataset', dataset + '.json')    #./dataset/vot2018.json
    annos = json.load(open(list_path, 'r'))
    seqs = annos.keys()


    

    trackers = glob.glob(fullfile('result', dataset, tracker_reg))
    trackers = trackers[start:min(end, len(trackers))]

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    # thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    # success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
    for i in range(n_seq):
        seq = list(seqs)[i]
        ground = readall(seq)
        gt_rect = np.array(ground).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
            bb = get_result_bb(tracker, seq)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
            # success_error[i][j] = compute_success_error(gt_center, center)

    print('Success Overlap')

    max_auc = 0.
    max_name = ''
    for i in range(len(trackers)):
        auc = success_overlap[:, i, :].mean()
        if auc > max_auc:
            max_auc = auc
            max_name = trackers[i]
        print('%s(%.4f)' % (trackers[i], auc))

    print('\n%s Best: %s(%.4f)' % (dataset, max_name, max_auc))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('python eval_vot.py LaSOT DCFNet_test* 0 10')
        exit()
    dataset = sys.argv[1]
    tracker_reg = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    eval_auc(dataset, tracker_reg, start, end)
