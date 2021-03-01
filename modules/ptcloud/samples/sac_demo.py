#!/usr/bin/env python
'''
SAC point cloud segmentation
==================

Example of using the ptcloud module

Usage
-----
sac_demo.py [ply file, model_type, threshold, max_iters]

'''

# Python 2/3 compatibility
from __future__ import print_function


import sys
import argparse
import numpy as np
import cv2 as cv

def run_plane():
	N = 64;
	plane = np.zeros((N*N,1,3),np.float32)
	for i in range(0,N):
	    for j in range(0,N):
	         plane[i*N+j] = (i,j,0)

	fit = cv.ptcloud.SACModelFitting_create(plane)
	mdl,left = fit.segment()
	print(mdl[0].coefficients.T)  # [[0,0,1,0]]
	print(mdl[0].score)           # (4096, -4096)
	print(len(mdl[0].points))     # 4096
	print(np.shape(left))         # ()

types= ["","plane","sphere","cylinder","cluster"]
def run_ply(fn, model_type=1, threshold=0.01, iters=10000):
	cloud = cv.ppf_match_3d.loadPLYSimple(fn)

	fit = cv.ptcloud.SACModelFitting_create(cloud, model_type, 1, threshold, iters)
	mdl,left = fit.segment()
	print(len(mdl), "models of type ", model_type, "found.")
	print(np.shape(left)[0], "points left")
	for i in range(len(mdl)):
		print(i, mdl[i].score, mdl[i].coefficients.T, len(mdl[i].points))
		cv.ppf_match_3d.writePLY(mdl[i].points, "sac_%s_%d.ply" % (types[model_type],i))


if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser(description='sac_demo: 3d point cloud segmentation')
    parser.add_argument('--ply', help='read a point cloud from a ply file')
    parser.add_argument('--threshold', help='threshold value', default=0.01)
    parser.add_argument('--model', help='which kind of model to segment', default=1)
    parser.add_argument('--iters', help='number of ransac iterations', default=1000)

    args = parser.parse_args()
    if (args.ply):
		run_ply(args.ply, int(args.model), float(args.threshold), int(args.iters))
	else:
		run_plane()
