'''
This example shows the using the Facemark API(fitting) for python.

USAGE: facemark_fit_demo.py <method - kazemi / aam / lbf > <path_to_input_image> <path to cascade classifier> <path to the pre-trained model>
'''

'''Please note - the above code will give a std::bad_alloc() error (due to the fit function) - please follow the
	this link to overcome the issue:
	https://github.com/opencv/opencv_contrib/issues/1661#issuecomment-397802106
'''

#example - python facemark_fit_demo.py lbf Messi.jpg haarcascade_frontalface_alt2.xml lbfmodel.yaml

import sys
import cv2 as cv

if __name__ == '__main__':

	if len(sys.argv) > 4:
		model_type = sys.argv[1]
		fname = sys.argv[2]
		cname = sys.argv[3]
		mname = sys.argv[4]

	else:
		print "Invalid number of parameters"
		print "Usage is: facemark_fit_demo.py <method - kazemi / aam / lbf > <path_to_input_image> <path to cascade classifier> <path to the pre-trained model>"
		sys.exit(1)

	img = cv.imread(fname)

	if img is None:
		print("Failed to load image file:", fname)
		sys.exit(1)

	#load the Cascade classifier
	cascade = cv.CascadeClassifier(cname)

	#store the faces detected in the target image
	faces = cascade.detectMultiScale(img)

	#creating a model based on the input given
	#default is LBF
	fm = cv.face.createFacemarkLBF()

	if(model_type == "aam"):
		fm = cv.face.createFacemarkAAM()
	elif model_type == "kazemi" :
		fm = cv.face.createFacemarkKazemi()

	#loading the pre-trained model
	fm.loadModel(mname)

	#running the algorithm and storing the landmarks found in the target image
	_, landmarks = fm.fit(img,faces)


	#looping over all the faces obtained and drawing landmarks on them
	#uses the drawFacemarks function
	for i in range(len(landmarks)):
		cv.face.drawFacemarks(img,landmarks[i])

	#display image
	cv.imshow('Image with landmark detection', img)
	cv.waitKey(0)
	cv.destroyAllWindows()
