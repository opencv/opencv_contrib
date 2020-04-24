import numpy as np
import cv2 as cv
import sys
import glob

#list_of_videos = [None for _ in range(280)]
list_of_videos = [None for _ in range(1)]
file_with_video_names = open("D:/LaSOTTesting/testing_set.txt", "r")

#for i in range(280):
for i in range(1):
    list_of_videos[i] = file_with_video_names.readline()

#trackers = [[None for _ in range(2)] for _ in range(7)]
#trackers = [[None for _ in range(2)] for _ in range(1)]

#trackers[0][0] = "boosting"
#trackers[1][0] = "mil"
#trackers[2][0] = "kcf"
#trackers[3][0] = "median_flow"
#trackers[4][0] = "goturn"
#trackers[5][0] = "mosse"
#trackers[6][0] = "csrt"

#trackers[0][1] = cv.TrackerBoosting_create()
#trackers[1][1] = cv.TrackerMIL_create()
#trackers[2][1] = cv.TrackerKCF_create()
#trackers[3][1] = cv.TrackerMedianFlow_create()
#trackers[4][1] = cv.TrackerGOTURN_create()
#trackers[5][1] = cv.TrackerMOSSE_create()
#trackers[6][1] = cv.TrackerCSRT_create()

trackers = [None for _ in range(7)]
trackers[0] = "Boosting"
trackers[1] = "MIL"
trackers[2] = "KCF"
trackers[3] = "MedianFlow"
trackers[4] = "GOTURN"
trackers[5] = "MOSSE"
trackers[6] = "CSRT"
#Cicle for trackers
for tracker_id in range(len(trackers)):
    tracker_name = trackers[tracker_id]

    #results = open("D:/LaSOT_benchmark_results.txt", "w")
    results = open("D:/results_" + tracker_name + ".txt", "w")
    #tracker = trackers[_][1]
    average_precisions = []
    #Cicle for videos
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

        print("Tracker: ", tracker)

        init_once = False

        print("Video: " + str(video_name))

        true_positive = 0
        false_positive = 0
        false_negative = 0

        print("Before evaluation:\nTrue Positive = " + str(true_positive) + " False Positive = " +
              str(false_positive) + " False Negative = " + str(false_negative))

        video_name = video_name.replace("\n", "")
        ground_truth = open("D:/LaSOTTesting/" +
                            video_name + "/groundtruth.txt", "r")
        ground_truth_bb = ground_truth.readline().replace("\n", "").split(",")
        init_bb = ground_truth_bb

        for _ in range(len(init_bb)):
            init_bb[_] = float(init_bb[_])

        init_bb = tuple(init_bb)

        print("Initial bounding box: ", init_bb)

        video_sequence = sorted(
            glob.glob(str("D:/LaSOTTesting/" + str(video_name) + "/img/*.jpg")))

        print("Number of frames: " + str(len(video_sequence)))
        #Cicle for frames
        for _, image in enumerate(video_sequence):
            #ground_truth_bb = ground_truth.readline().replace("\n", "").split(",")

            for _ in range(len(ground_truth_bb)):
                ground_truth_bb[_] = float(ground_truth_bb[_])

            ground_truth_bb = tuple(ground_truth_bb)

            frame = cv.imread(image)
            if not init_once:
                #print("frame ", frame)
                #print("init_bb ", init_bb)
                temp = tracker.init(frame, init_bb)
                init_once = True
                print("Status of initialization: ", temp)

            temp, new_bb = tracker.update(frame)
            #coordinates of points of bounding boxes
            new_bb_xmin = new_bb[0] - new_bb[2] / 2
            new_bb_xmax = new_bb[0] + new_bb[2] / 2
            new_bb_ymin = new_bb[1] - new_bb[3] / 2
            new_bb_ymax = new_bb[1] + new_bb[3] / 2
            ground_truth_xmin = ground_truth_bb[0] - ground_truth_bb[2] / 2
            ground_truth_xmax = ground_truth_bb[0] + ground_truth_bb[2] / 2
            ground_truth_ymin = ground_truth_bb[1] - ground_truth_bb[3] / 2
            ground_truth_ymax = ground_truth_bb[1] + ground_truth_bb[3] / 2
            #Width and height of overlap
            dx = min(new_bb_xmax, ground_truth_xmax) - \
                max(new_bb_xmin, ground_truth_xmin)
            dy = min(new_bb_ymax, ground_truth_ymax) - \
                max(new_bb_ymin, ground_truth_ymin)
            #Checking existing of intersection
            if (dx < 0) or (dy < 0):
                false_negative += 1
            else:
                #Area of Overlap
                AoO = dx * dy
                #Area of Union
                AoU = (new_bb_xmax - new_bb_xmin) * (new_bb_ymax - new_bb_ymin) + (
                    ground_truth_xmax - ground_truth_xmin) * (ground_truth_ymax - ground_truth_ymin) - AoO
                #Intersection over Union
                IoU = AoO / AoU
                #Threshold of success = 0.5
                if IoU > 0.5:
                    true_positive += 1
                elif IoU < 0.5 and IoU > 0:
                    false_positive += 1
            ground_truth_bb = ground_truth.readline().replace("\n", "").split(",")
        #Evaluation of average precision
        print("After evaluation :\nTrue Positive = " + str(true_positive) + " False Positive = " +
              str(false_positive) + " False Negative = " + str(false_negative) + "\n")
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        average_precision = precision * recall
        average_precisions.append(average_precision)
    #Evaluate mAP and write results in txt file
    mean_average_precision = sum(average_precisions) / len(average_precisions)
    results.write("Mean average precision of " + tracker_name +
                  " = " + str(mean_average_precision) + "\n")
#(opencv-contrib-3.4.2.17-pip)
