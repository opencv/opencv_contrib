"""
The benchmark for trackers in the "tracking" module of the "opencv-contrib"
repository. For evaluations used testing set of the LaSOT dataset:
https://cis.temple.edu/lasot/
For evaluations used metrics from LaSOT and TrackingNet papers.
Link to LaSOT paper: https://arxiv.org/abs/1809.07845
Link to TrackingNet paper: https://arxiv.org/abs/1803.10794
"""

import numpy as np
import cv2 as cv
import argparse
import os
import sys

path = cv.samples.findFile('samples/dnn/dasiamrpn_tracker.py')
sys.path.append(os.path.dirname(path))
from dasiamrpn_tracker import DaSiamRPNTracker

def get_iou(new, gt):
    '''
    During the calculation of intersection over union, we are checking
    numerical value of area_of_overlap, because if it is equal to 0,
    we have no intersection.
    '''
    new_xmin, new_ymin, new_w, new_h = new
    gt_xmin, gt_ymin, gt_w, gt_h = gt
    def get_max_coord(coord, size): return coord + size - 1.0
    new_xmax, new_ymax = get_max_coord(new_xmin, new_w), get_max_coord(
        new_ymin, new_h)
    gt_xmax, gt_ymax = get_max_coord(gt_xmin, gt_w), get_max_coord(
        gt_ymin, gt_h)
    dx = max(0, min(new_xmax, gt_xmax) - max(new_xmin, gt_xmin))
    dy = max(0, min(new_ymax, gt_ymax) - max(new_ymin, gt_ymin))
    area_of_overlap = dx * dy
    area_of_union = (new_xmax - new_xmin) * (new_ymax - new_ymin) + (
        gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - area_of_overlap
    iou = area_of_overlap / area_of_union if area_of_union != 0 else 0
    return iou


def get_pr(new, gt, is_norm):
    '''
    In calculations of precision and normalized precision are used thresholds
    from original TrackingNet paper. If third argument is "True" - we calculate
    normalized precision, if it is "False" - we calculate precision without
    normalization
    '''
    new_x, new_y, new_w, new_h = new
    gt_x, gt_y, gt_w, gt_h = gt
    def get_center(coord, size): return coord + (size + 1.0) / 2
    new_cx, new_cy, gt_cx, gt_cy = get_center(new_x, new_w), get_center(
        new_y, new_h), get_center(gt_x, gt_w), get_center(gt_y, gt_h)
    dx = new_cx - gt_cx
    dy = new_cy - gt_cy
    if is_norm:
        dx /= gt_w
        dy /= gt_h
    return np.sqrt(dx ** 2 + dy ** 2)


def init_tracker(tracker_name):
    '''
    Method used for initializing of trackers by creating it
    via cv.TrackerX_create().
    Input: string with tracker name.
    Output: dictionary 'config'
    Dictionary 'config' contains trackers names
    as keys and tuple with call method and number of frames for
    reinitialization as values.
    For evaluation of the DaSiamRPN tracker ONNX models should be placed
    in the same folder with Python script and benchmark
    '''
    config = {"Boosting": (cv.TrackerBoosting_create(), 1000),
              "MIL": (cv.TrackerMIL_create(), 1000),
              "KCF": (cv.TrackerKCF_create(), 1000),
              "MedianFlow": (cv.TrackerMedianFlow_create(), 1000),
              "GOTURN": (cv.TrackerGOTURN_create(), 250),
              "MOSSE": (cv.TrackerMOSSE_create(), 1000),
              "CSRT": (cv.TrackerCSRT_create(), 1000),
              "DaSiamRPN": (DaSiamRPNTracker(), 250)}
    return config[tracker_name]


def main():
    parser = argparse.ArgumentParser(
        description="Run LaSOT-based benchmark for visual object trackers")
    # As a default argument used name of the original dataset folder
    parser.add_argument("--dataset", type=str,
                        default="LaSOTTesting", help="Full path to LaSOT")
    parser.add_argument("-v", dest="visualization", action='store_true',
                        help="Showing process of tracking")
    args = parser.parse_args()

    # Creating list of names of the videos via reading names from the txt file
    video_names = os.path.join(args.dataset, "testing_set.txt")
    with open(video_names, 'rt') as f:
        list_of_videos = f.read().rstrip('\n').split('\n')
    trackers = [
        'Boosting', 'MIL', 'KCF', 'MedianFlow',\
        'GOTURN', 'MOSSE', 'CSRT', 'DaSiamRPN']

    iou_avg = []
    pr_avg = []
    n_pr_avg = []

    # Loop for every tracker
    for tracker_name in trackers:

        print("Tracker name: ", tracker_name)

        number_of_thresholds = 21
        iou_video = np.zeros(number_of_thresholds)
        pr_video = np.zeros(number_of_thresholds)
        n_pr_video = np.zeros(number_of_thresholds)
        iou_thr = np.linspace(0, 1, number_of_thresholds)
        pr_thr = np.linspace(0, 50, number_of_thresholds)
        n_pr_thr = np.linspace(0, 0.5, number_of_thresholds)

        # Loop for every video
        for video_name in list_of_videos:

            tracker, frames_for_reinit = init_tracker(tracker_name)
            init_once = False

            print("\tVideo name: " + str(video_name))

            # Open specific video and read ground truth for it
            gt_file = open(os.path.join(args.dataset, video_name,
                                        "groundtruth.txt"), "r")
            gt_bb = gt_file.readline().rstrip("\n").split(",")
            init_bb = tuple([float(b) for b in gt_bb])

            video_sequence = sorted(os.listdir(os.path.join(
                args.dataset, video_name, "img")))

            iou_values = []
            pr_values = []
            n_pr_values = []
            frame_counter = len(video_sequence)

            # Loop for every frame in the video
            for number_of_the_frame, image in enumerate(video_sequence):
                frame = cv.imread(os.path.join(
                    args.dataset, video_name, "img", image))
                gt_bb = tuple([float(x) for x in gt_bb])

                # Check for presence of the object on the image
                # Image is ignored if no object on it
                if gt_bb[2] == 0 or gt_bb[3] == 0:
                    gt_bb = gt_file.readline().rstrip("\n").split(",")
                    frame_counter -= 1
                    continue

                # Condition for reinitialization of the tracker
                if ((number_of_the_frame + 1) % frames_for_reinit == 0):
                    tracker, frames_for_reinit = init_tracker(tracker_name)
                    init_once = False
                    init_bb = gt_bb

                if not init_once:
                    init_state = tracker.init(frame, init_bb)
                    init_once = True
                init_state, new_bb = tracker.update(frame)

                if args.visualization:
                    new_x, new_y, new_w, new_h = list(map(int, new_bb))
                    cv.rectangle(frame, (new_x, new_y), ((
                        new_x + new_w), (new_y + new_h)), (200, 0, 0))
                    cv.imshow("Tracking", frame)
                    cv.waitKey(1)

                iou_values.append(get_iou(new_bb, gt_bb))
                pr_values.append(get_pr(new_bb, gt_bb, is_norm=False))
                n_pr_values.append(get_pr(new_bb, gt_bb, is_norm=True))

                # Setting as ground truth bounding box of the next frame
                gt_bb = gt_file.readline().rstrip("\n").split(",")

            # Calculating mean arithmetic value for the specific video
            iou_video += (np.fromiter([sum(
                i >= thr for i in iou_values).astype(
                    float) / frame_counter for thr in iou_thr], dtype=float))
            pr_video += (np.fromiter([sum(
                i <= thr for i in pr_values).astype(
                    float) / frame_counter for thr in pr_thr], dtype=float))
            n_pr_video += (np.fromiter([sum(
                i <= thr for i in n_pr_values).astype(
                    float) / frame_counter for thr in n_pr_thr], dtype=float))

        iou_mean_avg = np.array(iou_video) / len(list_of_videos)
        pr_mean_avg = np.array(pr_video) / len(list_of_videos)
        n_pr_mean_avg = np.array(n_pr_video) / len(list_of_videos)

        # We find the area under the curve according to the trapezoid rule
        # and normalize by the maximum threshold value
        iou = np.trapz(iou_mean_avg, x=iou_thr) / iou_thr[-1]
        pr = np.trapz(pr_mean_avg, x=pr_thr) / pr_thr[-1]
        n_pr = np.trapz(n_pr_mean_avg, x=n_pr_thr) / n_pr_thr[-1]

        iou_avg.append('%.4f' % iou)
        pr_avg.append('%.4f' % pr)
        n_pr_avg.append('%.4f' % n_pr)

    titles = ["Names:", "IoU:", "Precision:", "N.Precision:"]
    data = [titles] + list(zip(trackers, iou_avg, pr_avg, n_pr_avg))
    for number, for_tracker in enumerate(data):
        line = '|'.join(str(x).ljust(20) for x in for_tracker)
        print(line)
        if number == 0:
            print('-' * len(line))


if __name__ == '__main__':
    main()
