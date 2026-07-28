import cv2 as cv
import numpy as np
import time


def main():
    started = time.time()

    def feed():
        frame = np.full((240, 320, 3), (40, 120, 220), dtype=np.uint8)
        x = int(((time.time() - started) * 80) % frame.shape[1])
        cv.rectangle(frame, (x, 80), (min(x + 60, frame.shape[1] - 1), 150),
                     (240, 240, 80), -1)
        return frame

    view = cv.liveview.show(feed, name="sample", mode="auto", show=False)
    print("LiveView URL:", view.channel_url)
    input("Press Enter to stop.")
    view.close()


if __name__ == "__main__":
    main()
