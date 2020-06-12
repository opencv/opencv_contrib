function detect(img::OpenCV.InputArray, cascade)
    rects = OpenCV.detectMultiScale(cascade, img)
    return (rects[1].x, rects[1].y, rects[1].width+rects[1].x, rects[1].height+rects[1].y)
end


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

cascade = OpenCV.CascadeClassifier(joinpath(test_dir, "cascadeandhog", "cascades", "haarcascade_frontalface_alt.xml"))

img = OpenCV.imread(joinpath(test_dir, "cascadeandhog", "images", "mona-lisa.png"), OpenCV.IMREAD_GRAYSCALE)

rect = detect(img, cascade)

expected_rect = (164,119,306,261)

@test IOU(rect, expected_rect) > 0.95

print("objdetect test passed\n")
