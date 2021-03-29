# test simple blob detector
img_gray = OpenCV.imread(joinpath(test_dir, "shared", "pic1.png"), OpenCV.IMREAD_GRAYSCALE)

detector = OpenCV.SimpleBlobDetector_create()

# Compare centers of keypoints and se how many of them match,
kps = OpenCV.detect(detector, img_gray)

kps_expect = [OpenCV.Point{Float32}(174.9114f0, 227.75146f0),OpenCV.Point{Float32}(106.925545f0, 179.5765f0)]
for kp in kps
    closest_match = 100000
    for kpe in kps_expect
        dx = kpe.x - kp.pt.x
        dy = kpe.y - kp.pt.y
        if sqrt(dx*dx+dy*dy) < closest_match
            closest_match = sqrt(dx*dx+dy*dy)
        end
    end

    @test closest_match < 10
end

println("feature2d test passed")
