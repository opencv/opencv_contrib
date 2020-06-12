using OpenCV

function detect(img::OpenCV.InputArray, cascade)
    rects = OpenCV.detectMultiScale(cascade, img, scaleFactor=1.3, minNeighbors=Int32(4), minSize=OpenCV.Size{Int32}(30, 30), flags=OpenCV.CASCADE_SCALE_IMAGE)
    processed_rects = []
    for rect in rects
        push!(processed_rects, (rect.x, rect.y, rect.width+rect.x, rect.height+rect.y))
    end
    return processed_rects
end

function draw_rects(img, rects, color)
    for x in rects
        OpenCV.rectangle(img, OpenCV.Point{Int32}(x[1], x[2]), OpenCV.Point{Int32}(x[3], x[4]), color, thickness = Int32(2))
    end
end

cap = OpenCV.VideoCapture(Int32(0))

# Replace the paths for the classifiers before running

cascade = OpenCV.CascadeClassifier("haarcascade_frontalface_alt.xml")
nested = OpenCV.CascadeClassifier("haarcascade_eye.xml")

OpenCV.namedWindow("facedetect")

while true
    ret, img = OpenCV.read(cap)
    if ret==false
        print("Webcam stopped")
        break
    end
    gray = OpenCV.cvtColor(img, OpenCV.COLOR_BGR2GRAY)
    gray = OpenCV.equalizeHist(gray)

    rects = detect(gray, cascade)
    vis = copy(img)
    draw_rects(vis, rects, (0.0, 255.0, 0.0))

    if ~OpenCV.empty(nested)
        for x in rects
            roi = view(gray, :, Int(x[1]):Int(x[3]), Int(x[2]):Int(x[4]))
            subrects = detect(roi, nested)
            draw_view = view(vis, :, Int(x[1]):Int(x[3]), Int(x[2]):Int(x[4]))
            draw_rects(draw_view, subrects, (255.0, 0.0, 0.0))
        end
    end

    OpenCV.imshow("facedetect", vis)
    if OpenCV.waitKey(Int32(5))==27
        break
    end
end

OpenCV.release(cap)

OpenCV.destroyAllWindows()

print("Stopped")
