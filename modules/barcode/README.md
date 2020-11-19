1D Barcode Detect and Decode
======================

This module is focused on detecting and decoding barcode from image. It is mainly designed for scanning the images, locating barcode, decoding barcode and outputting its decoded result.

We provide two decoding methods, one-stage method and decode-after-detect method.

- One-stage method can decodes the barcodes in regular orientation. This module provides `decodeDirectly` as the port.
- Decode-after-detect method can locate the barcode with irregular orientation, then decode the cropped regions. However this method is limited by location accuracy, and due to detection method limitations, it can only detect medium-size barcodes.  This module provides `detectAndDecode` as the port, it also provides single step function `detect` and `decode`.
