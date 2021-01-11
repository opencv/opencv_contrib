WeChat QR code detector for detecting and parsing QR code.
================================================

WeChat QR code detector is a high-performance and lightweight QR code detect and decode library, which is contributed by WeChat Computer Vision Team (WeChatCV). It has been widely used in various Tencent applications, including WeChat, WeCom, QQ, QQ Browser, and so on. There are four primary features of WeChat QR code detector:

1. CNN-based QR code detector. Different from the traditional detector, we introduce a tiny CNN model for multiple code detection. The detector is based on SSD architecture with a MobileNetV2-like backbone, which is run on caffe inference framework.

2. CNN-based QR code enhancement. To improve the performance of tiny QR code, we design a lighten super-resolution CNN model for QR code, called QRSR. Depth-wise convolution, DenseNet concat and deconvolution are the core techniques in the QRSR model.

3. More robust finder pattern detection. Besides traditional horizontal line searching, we propose an area size based finder pattern detection method. we calculate the area size of black and white block to locate the finder pattern by the pre-computed connected cells.

4. Massive engineering optimization. Based on [zing-cpp](https://github.com/glassechidna/zxing-cpp), we conduct massive engineering optimization to boost the decoding success rate, such as trying more binarization methods, supporting N:1:3:1:1 finder pattern detection, finding more alignment pattern, clustering similar size finder pattern, and etc.
