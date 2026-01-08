import numpy as np
import cv2 as cv
import sys

from argparse import ArgumentParser

def get_depth_list(folder):
    f = open(folder + '/depth.txt', 'r')
    rgb = [folder + '/' + s for s in f.read().split() if s.endswith('.png')]
    return rgb

def kinfu_demo():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Required. Path to folder with a input image file", required=True, type=str)
    parser.add_argument(
        "-t", "--large_kinfu", help="Required. Name of KinFu type", required=False, type=str)
    parser.add_argument(
        "-ocl", "--use_opencl", help="Required. Flag of OpenCL use", required=False, type=int, default=1)

    args = parser.parse_args()
    print("Args: ", args)

    cv.ocl.setUseOpenCL(args.use_opencl)

    if (args.large_kinfu == None or args.large_kinfu == "0"):
        params = cv.kinfu_Params.defaultParams()
        kf = cv.kinfu_KinFu.create(params)
    elif (args.large_kinfu == "1"):
        params = cv.kinfu_Params.hashTSDFParams(False)
        kf = cv.kinfu_KinFu.create(params)
    else:
        raise ValueError("Incorrect kinfu type name")

    depth_list = get_depth_list(args.input)
    for path in depth_list:

        image = cv.imread(path, cv.IMREAD_ANYDEPTH)
        (height, width) = image.shape

        cv.imshow('input', image)

        size = height, width, 4
        cvt8 = np.zeros(size, dtype=np.uint8)

        if not kf.update(image):
            kf.reset()
        else:
            kf.render(cvt8)
            cv.imshow('render', cvt8)
        cv.pollKey()
    cv.waitKey(0)


if __name__ == '__main__':
    print(__doc__)
    kinfu_demo()
    cv.destroyAllWindows()
