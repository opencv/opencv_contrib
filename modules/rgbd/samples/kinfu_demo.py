import cv2
import numpy as np

from argparse import ArgumentParser

def get_depth_list(folder):
    if sys.platform.find('linux')+1:
        f = open(folder + '/depth.txt', 'r')
        rgb = [folder + '/' + s.replace('/', '/') for s in f.read().split() if '.png' in s]
    else:
        f = open(folder + '\\depth.txt', 'r')
        rgb = [folder + '\\' + s.replace('/', '\\') for s in f.read().split() if '.png' in s]
    return rgb

def kinfu_demo():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Required. Path to folder with a input image file", required=True, type=str)
    parser.add_argument(
        "-t", "--large_kinfu", help="Required. Name of KinFu type", required=False, type=str)
    
    args = parser.parse_args()
    print("Args: ", args)

    depth_list = get_depth_list(args.input)

    if (args.large_kinfu == None or args.large_kinfu == "0"):
        params = cv2.kinfu_Params.defaultParams()
        kf = cv2.kinfu_KinFu.create(params)
    elif (args.large_kinfu == "1"):
        params = cv2.kinfu_Params.hashTSDFParams(False)
        kf = cv2.kinfu_KinFu.create(params)
    else:
        raise ValueError("Incorrect kinfu type name")

    for path in depth_list:

        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        (height, width) = image.shape

        cv2.imshow('input', image)
        cv2.waitKey(1)


        size = height, width, 4
        cvt8 = np.zeros(size, dtype=np.uint8)

        #flag = kf.update(image)
        if not kf.update(image):
            kf.reset()
        else:
            kf.render(cvt8)
            cv2.imshow('render', cvt8)


if __name__ == '__main__':
    print(__doc__)
    cv2.setUseOptimized(True)
    kinfu_demo()
    cv2.destroyAllWindows()
