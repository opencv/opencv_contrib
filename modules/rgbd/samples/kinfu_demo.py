import cv2
import numpy as np

from argparse import ArgumentParser

def get_depth_list(folder):
    f = open(folder + '\\depth.txt', 'r')
    rgb = [folder + '\\' + s.replace('/', '\\') for s in f.read().split() if '.png' in s]
    return rgb

def kinfu_demo():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Required. Path to folder with a input image file", required=True, type=str)

    args = parser.parse_args()
    print("Args: ", args)
    depth_list = get_depth_list(args.input)
    params = cv2.kinfu_Params.defaultParams()
    kf = cv2.kinfu_KinFu.create(params)

    for path in depth_list:
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        (height, width) = image.shape[:]

        cv2.imshow('input', image)
        cv2.waitKey(1)

        size = height, width, 4
        cvt8 = np.zeros(size, dtype=np.uint8)

        flag = kf.update(image)
        if not flag:
            kf.reset()
        else:
            kf.render(cvt8)
            cv2.imshow('render', cvt8)


if __name__ == '__main__':
    print(__doc__)
    cv2.setUseOptimized(True)
    kinfu_demo()
    cv.destroyAllWindows()
