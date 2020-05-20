import numpy as np
import cv2 as cv
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description="Run LaSOT-based benchmark for visual object trackers")
parser.add_argument("--path_to_dataset", type=str,
                    default="Test part of LaSOT dataset", help="Full path to LaSOT folder")
args = parser.parse_args()

list_of_videos = [None for _ in range(280)]
file_with_video_names = open(args.path_to_dataset + "/testing_set.txt", "r")

for i in range(280):
    list_of_videos[i] = file_with_video_names.readline()

trackers = [None for _ in range(7)]
trackers = ["Boosting", "MIL", "KCF", "MedianFlow", "GOTURN", "MOSSE", "CSRT"]

#For every tracker
for tracker_id in range(len(trackers)):

    tracker_name = trackers[tracker_id]
    print(tracker_name)

    #For every video
    for video_name in list_of_videos:

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

        print("Video: " + str(video_name))

        video_name = video_name.replace("\n", "")
        gt_file = open(
            args.path_to_dataset + video_name + "/groundtruth.txt", "r")
        gt_bb = gt_file.readline().replace("\n", "").split(",")
        init_bb = gt_bb

        for _ in range(len(init_bb)):
            init_bb[_] = float(init_bb[_])

        init_bb = tuple(init_bb)

        print("Initial bounding box: ", init_bb)

        video_sequence = sorted(
            glob.glob(str(args.path_to_dataset + str(video_name) + "/img/*.jpg")))

        print("Number of frames in video: " + str(len(video_sequence)))

        sum_iou = 0
        sum_pr = 0
        sum_norm_pr = 0
        frame_counter = len(video_sequence)

        #For every frame in video
        for number_of_the_frame, image in enumerate(video_sequence):

            for _ in range(len(gt_bb)):
                gt_bb[_] = float(gt_bb[_])
            gt_bb = tuple(gt_bb)

            frame = cv.imread(image)

            #Re-initializing rate in frames
            if tracker_name == "Boosting":
                frames_before_reinit = 500
            elif tracker_name == "MIL":
                frames_before_reinit = 1000
            elif tracker_name == "KCF":
                frames_before_reinit = 110
            elif tracker_name == "MedianFlow":
                frames_before_reinit = 20
            elif tracker_name == "GOTURN":
                frames_before_reinit = 20
            elif tracker_name == "MOSSE":
                frames_before_reinit = 20
            elif tracker_name == "CSRT":
                frames_before_reinit = 250

            #Initialization and re-initialization of tracker
            if ((number_of_the_frame + 1) % frames_before_reinit == 0) and (
                number_of_the_frame != 0):

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
                init_bb = gt_bb
                first_frame = cv.imread(image)

            if not init_once:
                temp = tracker.init(frame, init_bb)
                init_once = True

            #Update tracker state
            temp, new_bb = tracker.update(frame)

            #Evaluation of coordinates of corners and centers from [x, y, w, h]
            new_bb_xmin = new_bb[0]
            new_bb_xmax = new_bb[0] + new_bb[2] - 1.0
            new_bb_ymin = new_bb[1]
            new_bb_ymax = new_bb[1] + new_bb[3] - 1.0
            gt_xmin = gt_bb[0]
            gt_xmax = gt_bb[0] + gt_bb[2] - 1.0
            gt_ymin = gt_bb[1]
            gt_ymax = gt_bb[1] + gt_bb[3] - 1.0
            new_cx = new_bb[0] + (new_bb[2] + 1.0) / 2
            new_cy = new_bb[1] + (new_bb[3] + 1.0) / 2
            gt_cx = gt_bb[0] + (gt_bb[2] + 1.0) / 2
            gt_cy = gt_bb[1] + (gt_bb[3] + 1.0) / 2

            # Width and height of overlap
            dx = max(0, min(new_bb_xmax, gt_xmax) - max(new_bb_xmin, gt_xmin))
            dy = max(0, min(new_bb_ymax, gt_ymax) - max(new_bb_ymin, gt_ymin))
            area_of_overlap = dx * dy
            area_of_union = (new_bb_xmax - new_bb_xmin) * (new_bb_ymax - new_bb_ymin) + (
                gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - area_of_overlap
            if (area_of_union != 0):
                # Intersection over Union
                intersection_over_union = area_of_overlap / area_of_union
                sum_iou += intersection_over_union

            precision = np.sqrt((new_cx - gt_cx) * (new_cx - gt_cx) + (new_cy - gt_cy) * (new_cy - gt_cy))
            if precision > 20.0:
                precision_value = 0.0
            else:
                precision_value = 1.0
            sum_pr += precision_value

            normalized_precision = np.sqrt((new_cx - gt_cx) / gt_bb[2] * (new_cx - gt_cx) / gt_bb[2] + (
                new_cy - gt_cy) / gt_bb[3] * (new_cy - gt_cy) / gt_bb[3])
            if normalized_precision < 0.5:
                normalized_precision_value = 0.0
            else:
                normalized_precision_value = 1.0
            sum_norm_pr += normalized_precision_value

            gt_bb = gt_file.readline().replace("\n", "").split(",")

        iou_values += sum_iou / frame_counter
        pr_values += sum_pr / frame_counter
        norm_pr_values += sum_norm_pr / frame_counter

    mean_iou = iou_values / 280
    mean_pr = pr_values / 280
    mean_norm_pr = norm_pr_values / 280

    print(tracker_name + ":\n\tmean IoU = " + str(mean_iou) + "\n\tmean precision = " + str(mean_pr) + "\n\tmean normalized precision = " + str(mean_norm_pr) + "\n\n")
