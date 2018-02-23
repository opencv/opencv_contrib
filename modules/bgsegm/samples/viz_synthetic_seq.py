import cv2 as cv
import argparse


def main():
    argparser = argparse.ArgumentParser(description='Vizualization of the SyntheticSequenceGenerator.')

    argparser.add_argument('-b', '--background', help='Background image.', required=True)
    argparser.add_argument('-o', '--obj', help='Object image. It must be strictly smaller than background.', required=True)
    args = argparser.parse_args()

    bg = cv.imread(args.background)
    obj = cv.imread(args.obj)
    generator = cv.bgsegm.createSyntheticSequenceGenerator(bg, obj)

    while True:
        frame, mask = generator.getNextFrame()
        cv.imshow('Generated frame', frame)
        cv.imshow('Generated mask', mask)
        k = cv.waitKey(int(1000.0 / 30))
        if k == 27:
            break


if __name__ == '__main__':
    main()
