import cv2 as cv


def main():
    view = cv.liveview.createServer()
    print(view)


if __name__ == "__main__":
    main()
