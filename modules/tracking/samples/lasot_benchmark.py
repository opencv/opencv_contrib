import numpy as np
import cv2 as cv
import argparse
import os
import glob

# Method used for evaluate corners coords 
# Not used yet - need to re-write calculations with it
def calc_coords(bb):
    xmin = bb[0]
    xmax = bb[0] + bb[2] - 1.0
    ymin = bb[1]
    ymax = bb[1] + bb[3] - 1.0
    cx = bb[0] + (bb[2] + 1.0) / 2
    cy = bb[1] + (bb[3] + 1.0) / 2
    coords = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                   'cx': cx, 'cy': cy}
    return coords

# Method used for initializing of trackers by creating it 
# via cv.TrackerX_create()
def init_tracker(tracker_name):
    config = {"Boosting": (cv.TrackerBoosting_create(), 500),
    "MIL": (cv.TrackerMIL_create(), 1000),
    "KCF": (cv.TrackerKCF_create(), 250),
    "MedianFlow": (cv.TrackerMedianFlow_create(), 500),
    "GOTURN": (cv.TrackerGOTURN_create(), 250),
    "MOSSE": (cv.TrackerMOSSE_create(), 250),
    "CSRT": (cv.TrackerCSRT_create(), 250)}
    return config[tracker_name]

def main():
    parser = argparse.ArgumentParser(
        description="Run LaSOT-based benchmark for visual object trackers")
    # As a default argument used name of 
    # original dataset folder
    parser.add_argument("--path_to_dataset", type=str,
                        default="LaSOTTesting", help="Full path to LaSOT")
    parser.add_argument("--visualization", type=str,
                        default=False, help="Showing process of tracking on video")
    args = parser.parse_args()

    # Creating list with names of videos via reading names from txt file

    video_names = os.path.join(args.path_to_dataset, "testing_set.txt")
    with open(video_names, 'rt') as f:
        list_of_videos = f.read().rstrip('\n').split('\n')

    trackers = [
        'Boosting', 'MIL', 'KCF', 'MedianFlow', 'GOTURN', 'MOSSE', 'CSRT']

    # Loop for every tracker
    for tracker_id in trackers:

        tracker_name = tracker_id
        print(tracker_name)

        # Loop for every video
        for video_name in list_of_videos:

            tracker, frames_before_reinit = init_tracker(tracker_name)
            init_once = False

            print("Video: " + str(video_name))

            # Open specific video and read ground truth for it
            gt_file = open(os.path.join(args.path_to_dataset, video_name, "groundtruth.txt"), "r")
            gt_bb = gt_file.readline().replace("\n", "").split(",")
            print(gt_bb)
            init_bb = gt_bb
            init_bb = tuple([float(b) for b in init_bb])

            print("Initial bounding box: ", init_bb)

            # Creating blob from image sequence
            #video_sequence = sorted(os.listdir(os.path.join(args.path_to_dataset, video_name, "img")))
            video_sequence = sorted(glob.glob(str("D:/lasot/" + str(video_name) +
                                                  "/img/*.jpg")))
            #print(video_sequence)
            print("Number of frames in video: " + str(len(video_sequence)))

            # Variables for saving sum of every metric for every frame and
            # every video respectively
            sum_iou = 0
            sum_pr = 0
            sum_norm_pr = 0
            iou_values = 0
            pr_values = 0
            norm_pr_values = 0
            frame_counter = len(video_sequence)

            # For every frame in video
            for number_of_the_frame, image in enumerate(video_sequence):

                for i in range(len(gt_bb)):
                    gt_bb[i] = float(gt_bb[i])
                gt_bb = tuple(gt_bb)
                frame = cv.imread(image)

                # Condition of tracker`s re-initialization
                if ((number_of_the_frame + 1) % frames_before_reinit == 0) and (
                        number_of_the_frame != 0):

                    tracker, frames_before_reinit = init_tracker(tracker_name)
                    init_once = False
                    init_bb = gt_bb

                if not init_once:
                    init_state = tracker.init(frame, init_bb)
                    init_once = True
                init_state, new_bb = tracker.update(frame)

                # Check for presence of object on the frame
                # If no object on frame, we reduce number of
                # accounted for evaluation frames
                if gt_bb != (0, 0, 0, 0):
                    # Calculation of coordinates of corners and centers 
                    # from [x, y, w, h] bounding boxes

                    new_coords = calc_coords(new_bb)
                    gt_coords = calc_coords(gt_bb)
                    new_xmin, new_xmax, new_ymin, new_ymax, new_cx, new_cy = list(
                        new_coords.values())
                    gt_xmin, gt_xmax, gt_ymin, gt_ymax, gt_cx, gt_cy = list(gt_coords.values())

                    if args.visualization:
                        cv.rectangle(frame, (int(new_xmin), int(new_ymin)), (
                            int(new_xmax), int(new_ymax)), (200, 0, 0))
                        cv.imshow("Tracking", frame)
                        cv.waitKey(1)
                    # Metrics: Intersection over Union, Precision, 
                    # Normalized Precision

                    # Width and height of overlap
                    dx = max(0, min(new_xmax, gt_xmax) - max(
                        new_xmin, gt_xmin))
                    dy = max(0, min(new_ymax, gt_ymax) - max(
                        new_ymin, gt_ymin))

                    # Intersection over Union
                    area_of_overlap = dx * dy
                    area_of_union = (new_xmax - new_xmin) * (
                        new_ymax - new_ymin) + (gt_xmax - gt_xmin) * (
                            gt_ymax - gt_ymin) - area_of_overlap
                    # Zero division check
                    if area_of_union != 0:
                        iou = area_of_overlap / area_of_union
                        sum_iou += iou

                    # Precision
                    precision = np.sqrt((new_cx - gt_cx) ** 2 + (
                        new_cy - gt_cy) ** 2)
                    if precision > 20.0:
                        pr_value = 0.0
                    else:
                        pr_value = 1.0
                    sum_pr += pr_value

                    # Normalized precision
                    # Zero division check
                    if gt_bb[2] != 0 and gt_bb[3] != 0:
                        normalized_precision = np.sqrt(
                            ((new_cx - gt_cx) / gt_bb[2]) ** 2 + (
                                (new_cy - gt_cy) / gt_bb[3]) ** 2)
                        if normalized_precision > 0.1:
                            norm_pr_value = 0.0
                        else:
                            norm_pr_value = 1.0
                        sum_norm_pr += norm_pr_value
                else:
                    frame_counter -= 1

                # Setting as ground truth bounding box from next frame
                gt_bb = gt_file.readline().replace("\n", "").split(",")

            # Calculating mean arithmetic value for specific video
            iou_values += sum_iou / frame_counter
            pr_values += sum_pr / frame_counter
            norm_pr_values += sum_norm_pr / frame_counter

        # Calculating mean arithmetic value for dataset
        mean_iou = iou_values / len(list_of_videos)
        mean_pr = pr_values / len(list_of_videos)
        mean_norm_pr = norm_pr_values / len(list_of_videos)

        print(tracker_name + ":")
        print("\n\t" + str("Mean Intersection over Union = " + str(mean_iou)))
        print("\n\t" + str("Mean Precision = ") + str(mean_pr))
        print("\n\t" + str("Mean Normalized Precision = ") + str(mean_norm_pr))
        print("\n\n")

if __name__ == '__main__':
    main()
