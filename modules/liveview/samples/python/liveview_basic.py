import cv2 as cv
import numpy as np


def main():
    view = cv.liveview.createServer("127.0.0.1", 0)
    view.start()
    frame = np.full((240, 320, 3), (40, 120, 220), dtype=np.uint8)
    view.publish("sample", frame)
    print("LiveView URL:", view.url())
    input("Press Enter to stop.")


if __name__ == "__main__":
    main()
