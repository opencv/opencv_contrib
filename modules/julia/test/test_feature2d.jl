# test simple blob detector

img = OpenCV.imread(img_dir)
img_gray = OpenCV.cvtColor(img, OpenCV.COLOR_BGR2GRAY)

detector = OpenCV.SimpleBlobDetector_create()
kps = OpenCV.detect(detector, img_gray)

kps_expect = [] 
println("Number of keypoints: ", size(kps))
for kp in kps
    println(kp.pt, "\t", kp.size)
end