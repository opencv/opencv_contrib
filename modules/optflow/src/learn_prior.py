import os, sys
import numpy as np
import cv2
import struct
from math import sqrt

basis_size = (14, 18)
lambd = 0.2

def find_flo(pp):
	f = []
	for p in pp:
		if os.path.isfile(p):
			f.append(p)
		else:
			for root, subdirs, files in os.walk(p):
				f += map(lambda x: os.path.join(root, x), filter(lambda x: x.split('.')[-1] == 'flo', files))
	return list(set(f))

def load_flo(flo):
    with open(flo) as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                print('Reading %dx%d flo file %s' % (w, h, flo))
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                flow = np.resize(data, (h, w, 2))
                return flow[:,:,0], flow[:,:,1]

def get_w(m):
	s = m.shape
	w = cv2.dct(m)
	w *= 2.0 / sqrt(s[0] * s[1])
	#w[0,0] *= 0.5
	w[:,0] *= sqrt(0.5)
	w[0,:] *= sqrt(0.5)
	w = w[0:basis_size[0],0:basis_size[1]].transpose().flatten()
	return w

w1 = []
w2 = []

for flo in find_flo(sys.argv[1:]):
	x,y = load_flo(flo)
	w1.append(get_w(x))
	w2.append(get_w(y))

w1mean = sum(w1) / len(w1)
w2mean = sum(w2) / len(w2)

for i in xrange(len(w1)):
	w1[i] -= w1mean
for i in xrange(len(w2)):
	w2[i] -= w2mean

Q1 = sum([w1[i].reshape(-1,1).dot(w1[i].reshape(1,-1)) for i in xrange(len(w1))]) / len(w1)
Q2 = sum([w2[i].reshape(-1,1).dot(w2[i].reshape(1,-1)) for i in xrange(len(w2))]) / len(w2)
Q1 = np.matrix(Q1)
Q2 = np.matrix(Q2)

if len(w1) > 1:
	while True:
		try:
			L1 = np.linalg.cholesky(Q1)
			break
		except np.linalg.linalg.LinAlgError:
			mev = min(np.linalg.eig(Q1)[0]).real
			assert(mev < 0)
			print('Q1', mev)
			if -mev < 1e-6:
				mev = -1e-6
			Q1 += (-mev*1.000001) * np.identity(Q1.shape[0])

	while True:
		try:
			L2 = np.linalg.cholesky(Q2)
			break
		except np.linalg.linalg.LinAlgError:
			mev = min(np.linalg.eig(Q2)[0]).real
			assert(mev < 0)
			print('Q2', mev)
			if -mev < 1e-6:
				mev = -1e-6
			Q2 += (-mev*1.000001) * np.identity(Q2.shape[0])
else:
	L1 = np.identity(Q1.shape[0])
	L2 = np.identity(Q2.shape[0])

L1 = np.linalg.inv(L1) * lambd
L2 = np.linalg.inv(L2) * lambd

assert(L1.shape == L2.shape)
assert(L1.shape[0] == L1.shape[1])

f = open('cov.dat', 'wb')

f.write(struct.pack('I', L1.shape[0]))
f.write(struct.pack('I', L1.shape[1]))

for i in xrange(L1.shape[0]):
	for j in xrange(L1.shape[1]):
		f.write(struct.pack('f', L1[i,j])) 

for i in xrange(L2.shape[0]):
	for j in xrange(L2.shape[1]):
		f.write(struct.pack('f', L2[i,j]))

b1 = L1.dot(w1mean.reshape(-1,1))
b2 = L2.dot(w2mean.reshape(-1,1))

assert(L1.shape[0] == b1.shape[0])

for i in xrange(b1.shape[0]):
	f.write(struct.pack('f', b1[i,0]))

for i in xrange(b2.shape[0]):
	f.write(struct.pack('f', b2[i,0]))

f.close()