import numpy as np
import cv2 as cv
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description="Run benchmark")
parser.add_argument("--path_to_dataset", type=str, default="LaSOT", help="Full path to LaSOT folder")
parser.add_argument("--path_to_results", type=str, default="results.txt", help="Full path to file with results")
args = parser.parse_args()

list_of_videos = [None for _ in range(280)]
file_with_video_names = open(args.path_to_dataset + "/testing_set.txt", "r")

results = open(args.path_to_results, "w")

for i in range(280):
    list_of_videos[i] = file_with_video_names.readline()

trackers = [None for _ in range(6)]
#trackers = [None for _ in range(7)]
trackers[0] = "MedianFlow"
trackers[1] = "MIL"
trackers[2] = "CSRT"
trackers[3] = "Boosting"
trackers[4] = "KCF"
trackers[5] = "MOSSE"
#trackers[6] = "GOTURN"


# Loop for trackers
for tracker_id in range(len(trackers)):
    tracker_name = trackers[tracker_id]

    average_precisions = []
    # Loop for videos
    for video_name in list_of_videos:

        if tracker_name == "Boosting":
            tracker = cv.TrackerBoosting_create()
        elif tracker_name == "MIL":
            tracker = cv.TrackerMIL_create()
        elif tracker_name == "KCF":
            tracker = cv.TrackerKCF_create()
        elif tracker_name == "MedianFlow":
            tracker = cv.TrackerMedianFlow_create()
        elif tracker_name == "MOSSE":
            tracker = cv.TrackerMOSSE_create()
        elif tracker_name == "CSRT":
            tracker = cv.TrackerCSRT_create()
        elif tracker_name == "GOTURN":
            tracker = cv.TrackerGOTURN_create()

        print("Tracker: ", tracker)

        init_once = False

        print("Video: " + str(video_name))

        true_positive = 0
        false_positive = 0
        false_negative = 0

        video_name = video_name.replace("\n", "")
        ground_truth = open(args.path_to_dataset +
                            video_name + "/groundtruth.txt", "r")
        ground_truth_bb = ground_truth.readline().replace("\n", "").split(",")
        init_bb = ground_truth_bb

        for _ in range(len(init_bb)):
            init_bb[_] = float(init_bb[_])

        init_bb = tuple(init_bb)

        video_sequence = sorted(
            glob.glob(str(args.path_to_dataset + str(video_name) + "/img/*.jpg")))

        print("Number of frames: " + str(len(video_sequence)))
        # Loop for frames
        for f, image in enumerate(video_sequence):

            for _ in range(len(ground_truth_bb)):
                ground_truth_bb[_] = float(ground_truth_bb[_])

            ground_truth_bb = tuple(ground_truth_bb)

            frame = cv.imread(image)

            if ((f + 1) % 100 == 0) and (f != 0):
                if tracker_name == "Boosting":
                    tracker = cv.TrackerBoosting_create()
                elif tracker_name == "MIL":
                    tracker = cv.TrackerMIL_create()
                elif tracker_name == "KCF":
                    tracker = cv.TrackerKCF_create()
                elif tracker_name == "MedianFlow":
                    tracker = cv.TrackerMedianFlow_create()
                elif tracker_name == "GOTURN":
                    tracker = cv.TrackerGOTURN_create()
                elif tracker_name == "MOSSE":
                    tracker = cv.TrackerMOSSE_create()
                elif tracker_name == "CSRT":
                    tracker = cv.TrackerCSRT_create()
                init_once = False
                init_bb = new_bb

            if not init_once:
                temp = tracker.init(frame, init_bb)
                init_once = True

            temp, new_bb = tracker.update(frame)
            # Coordinates of points of bounding boxes
            new_bb_xmin = new_bb[0]
            new_bb_xmax = new_bb[0] + new_bb[2]
            new_bb_ymin = new_bb[1]
            new_bb_ymax = new_bb[1] + new_bb[3]
            ground_truth_xmin = ground_truth_bb[0]
            ground_truth_xmax = ground_truth_bb[0] + ground_truth_bb[2] 
            ground_truth_ymin = ground_truth_bb[1]
            ground_truth_ymax = ground_truth_bb[1] + ground_truth_bb[3]
            # Width and height of overlap
            dx = min(new_bb_xmax, ground_truth_xmax) - \
                max(new_bb_xmin, ground_truth_xmin)
            dy = min(new_bb_ymax, ground_truth_ymax) - \
                max(new_bb_ymin, ground_truth_ymin)
            # Checking existance of intersection
            if (dx < 0) or (dy < 0):
                false_negative += 1
            else:
                # Area of Overlap
                AoO = dx * dy
                # Area of Union
                AoU = (new_bb_xmax - new_bb_xmin) * (new_bb_ymax - new_bb_ymin) + (
                    ground_truth_xmax - ground_truth_xmin) * (ground_truth_ymax - ground_truth_ymin) - AoO
                if (AoU != 0):
                    # Intersection over Union
                    IoU = AoO / AoU
                    # Threshold of success = 0.5
                    if IoU > 0.5:
                        true_positive += 1
                    elif IoU < 0.5 and IoU > 0:
                        false_positive += 1
            ground_truth_bb = ground_truth.readline().replace("\n", "").split(",")
        # Evaluation of average precision
        print("After evaluation :\nTrue Positive = " + str(true_positive) + " False Positive = " +
              str(false_positive) + " False Negative = " + str(false_negative) + "\n")
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        average_precision = precision * recall
        average_precisions.append(average_precision)
    # Evaluate mAP and write results in txt file
    mean_average_precision = sum(average_precisions) / len(average_precisions)
    results.write("Mean average precision of " + tracker_name +
                  " = " + str(mean_average_precision) + "\n")
