import numpy as np
import cv2
import argparse
import os


def main():
    argparser = argparse.ArgumentParser(description='Vizualization of the LSBP/GSOC background subtraction algorithm.')

    argparser.add_argument('-g', '--gt', help='Directory with ground-truth frames', required=True)
    argparser.add_argument('-f', '--frames', help='Directory with input frames', required=True)
    argparser.add_argument('-l', '--lsbp', help='Display LSBP instead of GSOC', default=False)
    args = argparser.parse_args()

    gt = map(lambda x: os.path.join(args.gt, x), os.listdir(args.gt))
    gt.sort()
    f = map(lambda x: os.path.join(args.frames, x), os.listdir(args.frames))
    f.sort()

    gt = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), gt))
    f = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), f))

    if not args.lsbp:
        bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()
    else:
        bgs = cv2.bgsegm.createBackgroundSubtractorLSBP()

    for i in xrange(f.shape[0]):
        cv2.imshow('Frame', f[i])
        cv2.imshow('Ground-truth', gt[i])
        mask = bgs.apply(f[i])
        bg = bgs.getBackgroundImage()
        cv2.imshow('BG', bg)
        cv2.imshow('Output mask', mask)
        k = cv2.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
