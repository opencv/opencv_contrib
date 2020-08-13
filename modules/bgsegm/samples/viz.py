from builtins import range
import numpy as np
import cv2 as cv
import argparse
import os


def main():
    argparser = argparse.ArgumentParser(description='Vizualization of the LSBP/GSOC background subtraction algorithm.')

    argparser.add_argument('-g', '--gt', help='Directory with ground-truth frames', required=True)
    argparser.add_argument('-f', '--frames', help='Directory with input frames', required=True)
    argparser.add_argument('-l', '--lsbp', help='Display LSBP instead of GSOC', default=False)
    args = argparser.parse_args()

    gt = [os.path.join(args.gt, x) for x in os.listdir(args.gt)]
    gt.sort()
    f = [os.path.join(args.frames, x) for x in os.listdir(args.frames)]
    f.sort()

    gt = np.uint8([cv.imread(x, cv.IMREAD_GRAYSCALE) for x in gt])
    f = np.uint8([cv.imread(x, cv.IMREAD_COLOR) for x in f])

    if not args.lsbp:
        bgs = cv.bgsegm.createBackgroundSubtractorGSOC()
    else:
        bgs = cv.bgsegm.createBackgroundSubtractorLSBP()

    for i in range(f.shape[0]):
        cv.imshow('Frame', f[i])
        cv.imshow('Ground-truth', gt[i])
        mask = bgs.apply(f[i])
        bg = bgs.getBackgroundImage()
        cv.imshow('BG', bg)
        cv.imshow('Output mask', mask)
        k = cv.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
