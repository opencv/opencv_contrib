import cv2
import imutils
import os

def extract_frames_from_video(video_path):
    # Extract the video name and create output directory
    vid = os.path.basename(video_path)
    name = "/" + vid[:-4] + "-"
    write_directory = vid[:-4]

    try:
        os.mkdir(write_directory)
    except FileExistsError:
        print("Folder already exists.")

    # Read the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image
        cv2.imwrite(f"{write_directory}{name}{frame_count}.jpg", frame)

        # Resize frame and show
        frame = imutils.resize(frame, height=720)
        cv2.imshow('Frame', frame)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {frame_count} frames from {video_path}")
