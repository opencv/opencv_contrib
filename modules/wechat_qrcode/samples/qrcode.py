import cv2
import sys

print(sys.argv[0])
print('A demo program of WeChat QRCode Detector:')
camIdx = -1
if len(sys.argv) > 1:
    if sys.argv[1] == "-camera":
        camIdx = int(sys.argv[2]) if len(sys.argv)>2 else 0
    img = cv2.imread(sys.argv[1])
else:
    print("    Usage: " + sys.argv[0] + "  <input_image>")
    exit(0)

# For python API generator, it follows the template: {module_name}_{class_name},
# so it is a little weird.
# Provide paths to the detector and super-resolution ONNX models.
# Download links are provided in the error message below.
try:
    detector = cv2.wechat_qrcode_WeChatQRCode(
        "detect.onnx", "sr.onnx")

except:
    print("---------------------------------------------------------------")
    print("Failed to initialize WeChatQRCode.")
    print("Please download the ONNX models from:")
    print("Detector: https://github.com/omrope79/opencv-test-models/releases/download/v1.1.0/detect.onnx")
    print("Super-resolution: https://github.com/omrope79/opencv-test-models/releases/download/v1.1.0/sr.onnx")
    print("Ensure the model paths passed to WeChatQRCode are correct.")
    print("---------------------------------------------------------------")
    exit(0)


prevstr = ""

if camIdx < 0:
    res, points = detector.detectAndDecode(img)
    print(res,points)
else:
    cap = cv2.VideoCapture(camIdx)
    while True:
        res, img = cap.read()
        if img is None:
            break
        res, points = detector.detectAndDecode(img)
        for t in res:
            if t != prevstr:
                print(t)
        if res:
            prevstr = res[-1]
        cv2.imshow("image", img)
        if cv2.waitKey(30) >= 0:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
