import argparse
from framextract import extract_frames_from_video

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video
    extract_frames_from_video(video_path)

if __name__ == "__main__":
    main()
