function IOU(boxA, boxB)
	xA = max(boxA[1], boxB[1])
	yA = max(boxA[2], boxB[2])
	xB = min(boxA[3], boxB[3])
	yB = min(boxA[4], boxB[4])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[4] - boxA[2] + 1)
	boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[4] - boxB[2] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou
end

const cv = OpenCV
net = cv.dnn.DetectionModel(joinpath(ENV["OPENCV_TEST_DATA_PATH"], "dnn", "opencv_face_detector.pbtxt"),joinpath(ENV["OPENCV_TEST_DATA_PATH"], "dnn", "opencv_face_detector_uint8.pb"))
size0 = 300

cv.dnn.setPreferableTarget(net, cv.dnn.DNN_TARGET_CPU)
cv.dnn.setInputMean(net, (104, 177, 123))
cv.dnn.setInputScale(net, 1.)
cv.dnn.setInputSize(net, size0, size0)


img = OpenCV.imread(joinpath(test_dir, "cascadeandhog", "images", "mona-lisa.png"))

classIds, confidences, boxes = cv.dnn.detect(net, img, confThreshold=0.5)

box = (boxes[1].x, boxes[1].y, boxes[1].x+boxes[1].width, boxes[1].y+boxes[1].height)
expected_rect = (185,101,129+185,169+101)

@test IOU(box, expected_rect) > 0.8

print("dnn test passed\n")