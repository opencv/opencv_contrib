import cv2
import argparse


def main():
    argparser = argparse.ArgumentParser(description='Vizualization of the SyntheticSequenceGenerator.')

    argparser.add_argument('-b', '--background', help='Background image.', required=True)
    argparser.add_argument('-o', '--obj', help='Object image. It must be strictly smaller than background.', required=True)
    args = argparser.parse_args()

    bg = cv2.imread(args.background)
    obj = cv2.imread(args.obj)
    generator = cv2.bgsegm.createSyntheticSequenceGenerator(bg, obj)

    while True:
        frame, mask = generator.getNextFrame()
        cv2.imshow('Generated frame', frame)
        cv2.imshow('Generated mask', mask)
        k = cv2.waitKey(int(1000.0 / 30))
        if k == 27:
            break


if __name__ == '__main__':
    main()
