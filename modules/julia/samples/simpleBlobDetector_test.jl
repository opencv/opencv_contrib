using OpenCV

println("")
println("This is a simple exmample demonstrating the use of SimpleBlobDetector")
println("")
print("Path to image: ")
img_dir = readline()

img = OpenCV.imread(img_dir)
img_gray = OpenCV.cvtColor(img, OpenCV.COLOR_BGR2GRAY)

OpenCV.namedWindow("Img - Color")
OpenCV.namedWindow("Img - Gray")


OpenCV.imshow("Img - Color", img)
OpenCV.imshow("Img - Gray", img_gray)

OpenCV.waitKey(Int32(0))

OpenCV.destroyAllWindows()

detector = OpenCV.SimpleBlobDetector_create()
kps = OpenCV.detect(detector, img_gray)

println("Number of keypoints: ", size(kps))
for kp in kps
    println(kp.pt, "\t", kp.size)
end
