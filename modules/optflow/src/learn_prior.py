#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2 as cv
import struct
import argparse
from math import sqrt

argparser = argparse.ArgumentParser(
    description='''Use this script to generate prior for using with PCAFlow.
Basis size here must match corresponding parameter in the PCAFlow.
Gamma should be selected experimentally.''')

argparser.add_argument('-f',
                       '--files',
                       nargs='+',
                       help='List of optical flow .flo files for learning. You can pass a directory here and it will be scanned recursively for .flo files.',
                       required=True)
argparser.add_argument('-o',
                       '--output',
                       help='Output file for prior',
                       required=True)
argparser.add_argument('--width',
                       type=int,
                       help='Size of the basis first dimension',
                       required=True,
                       default=18)
argparser.add_argument('--height',
                       type=int,
                       help='Size of the basis second dimension',
                       required=True,
                       default=14)
argparser.add_argument(
    '-g',
    '--gamma',
    type=float,
    help='Amount of regularization. The greater this parameter, the bigger will be an impact of the regularization.',
    required=True)
args = argparser.parse_args()

basis_size = (args.height, args.width)
gamma = args.gamma


def find_flo(pp):
    f = []
    for p in pp:
        if os.path.isfile(p):
            f.append(p)
        else:
            for root, subdirs, files in os.walk(p):
                f += map(lambda x: os.path.join(root, x),
                         filter(lambda x: x.split('.')[-1] == 'flo', files))
    return list(set(f))


def load_flo(flo):
    with open(flo, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print('Reading %dx%d flo file %s' % (w, h, flo))
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.reshape(data, (h, w, 2))
            return flow[:, :, 0], flow[:, :, 1]


def get_w(m):
    s = m.shape
    w = cv.dct(m)
    w *= 2.0 / sqrt(s[0] * s[1])
    #w[0,0] *= 0.5
    w[:, 0] *= sqrt(0.5)
    w[0, :] *= sqrt(0.5)
    w = w[0:basis_size[0], 0:basis_size[1]].transpose().flatten()
    return w


w1 = []
w2 = []

for flo in find_flo(args.files):
    x, y = load_flo(flo)
    w1.append(get_w(x))
    w2.append(get_w(y))

w1mean = sum(w1) / len(w1)
w2mean = sum(w2) / len(w2)

for i in xrange(len(w1)):
    w1[i] -= w1mean
for i in xrange(len(w2)):
    w2[i] -= w2mean

Q1 = sum([w1[i].reshape(-1, 1).dot(w1[i].reshape(1, -1))
          for i in xrange(len(w1))]) / len(w1)
Q2 = sum([w2[i].reshape(-1, 1).dot(w2[i].reshape(1, -1))
          for i in xrange(len(w2))]) / len(w2)
Q1 = np.matrix(Q1)
Q2 = np.matrix(Q2)

if len(w1) > 1:
    while True:
        try:
            L1 = np.linalg.cholesky(Q1)
            break
        except np.linalg.linalg.LinAlgError:
            mev = min(np.linalg.eig(Q1)[0]).real
            assert (mev < 0)
            print('Q1', mev)
            if -mev < 1e-6:
                mev = -1e-6
            Q1 += (-mev * 1.000001) * np.identity(Q1.shape[0])

    while True:
        try:
            L2 = np.linalg.cholesky(Q2)
            break
        except np.linalg.linalg.LinAlgError:
            mev = min(np.linalg.eig(Q2)[0]).real
            assert (mev < 0)
            print('Q2', mev)
            if -mev < 1e-6:
                mev = -1e-6
            Q2 += (-mev * 1.000001) * np.identity(Q2.shape[0])
else:
    L1 = np.identity(Q1.shape[0])
    L2 = np.identity(Q2.shape[0])

L1 = np.linalg.inv(L1) * gamma
L2 = np.linalg.inv(L2) * gamma

assert (L1.shape == L2.shape)
assert (L1.shape[0] == L1.shape[1])

f = open(args.output, 'wb')

f.write(struct.pack('I', L1.shape[0]))
f.write(struct.pack('I', L1.shape[1]))

for i in xrange(L1.shape[0]):
    for j in xrange(L1.shape[1]):
        f.write(struct.pack('f', L1[i, j]))

for i in xrange(L2.shape[0]):
    for j in xrange(L2.shape[1]):
        f.write(struct.pack('f', L2[i, j]))

b1 = L1.dot(w1mean.reshape(-1, 1))
b2 = L2.dot(w2mean.reshape(-1, 1))

assert (L1.shape[0] == b1.shape[0])

for i in xrange(b1.shape[0]):
    f.write(struct.pack('f', b1[i, 0]))

for i in xrange(b2.shape[0]):
    f.write(struct.pack('f', b2[i, 0]))

f.close()
