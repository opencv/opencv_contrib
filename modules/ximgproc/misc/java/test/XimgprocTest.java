package org.opencv.test.ximgproc;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.test.OpenCVTestCase;
import org.opencv.ximgproc.Ximgproc;

public class XimgprocTest extends OpenCVTestCase {

    public void testHoughPoint2Line() {
        Mat src = new Mat(80, 80, CvType.CV_8UC1, new org.opencv.core.Scalar(0));
        Point houghPoint = new Point(40, 40);

        int[] result = Ximgproc.HoughPoint2Line(houghPoint, src, Ximgproc.ARO_315_135, Ximgproc.HDO_DESKEW, Ximgproc.RO_IGNORE_BORDERS);

        assertNotNull(result);
        assertEquals(4, result.length);
    }
}
