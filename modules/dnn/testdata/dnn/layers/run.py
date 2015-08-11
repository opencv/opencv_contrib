# coding: utf-8

import sys, os, glob

CAFFE_ROOT = "/home/vitaliy/opencv/caffe/"
sys.path.insert(0, CAFFE_ROOT + 'python')

CV2_DIR = "/home/vitaliy/opencv/build-opencv-qt/lib"
sys.path.insert(0, CV2_DIR)


import numpy as np
import caffe
import cv2

def get_cafe_output(inp_blob, proto_name, caffemodel_name):
    caffe.set_mode_cpu()
    net = caffe.Net(proto_name, caffe.TEST)

    net.blobs['input'].reshape(*inp_blob.shape)
    net.blobs['input'].data[...] = inp_blob

    net.forward()
    out_blob = net.blobs['output'].data[...];

    if net.params.get('output'):
        print "Params count:", len(net.params['output'])
        net.save(caffemodel_name)

    return out_blob

if __name__ == '__main__':
    proto_filenames = glob.glob("*.prototxt")

    inp_blob = np.load('blob.npy')
    print inp_blob.shape

    for proto_filename in proto_filenames:
        proto_filename = os.path.basename(proto_filename)
        proto_basename = os.path.splitext(proto_filename)[0]
        cfmod_basename = proto_basename + ".caffemodel"
        npy_filename = proto_basename + ".npy"

        print cfmod_basename

        out_blob = get_cafe_output(inp_blob, proto_filename, cfmod_basename)
        print out_blob.shape
        np.save(npy_filename, out_blob)
