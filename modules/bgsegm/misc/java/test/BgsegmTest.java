package org.opencv.test.bgsegm;

import org.opencv.bgsegm.*;
import org.opencv.bgsegm.Bgsegm.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

public class BgsegmTest extends OpenCVTestCase {
    Mat img;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        img = new Mat(300,300,CvType.CV_8U);
    }

    public void testCNT() {
        BackgroundSubtractorCNT bgs = Bgsegm.createBackgroundSubtractorCNT();
        assertNotNull("could not create a CNT instance!", bgs);
        Mat mask = new Mat();
        bgs.apply(img,mask);
        assertFalse("no mask created from CNT", mask.empty());
    }
    public void testMOG() {
        BackgroundSubtractorMOG bgs = Bgsegm.createBackgroundSubtractorMOG();
        assertNotNull("could not create a MOG instance!", bgs);
        Mat mask = new Mat();
        bgs.apply(img,mask);
        assertFalse("no mask created from MOG", mask.empty());
    }
    public void testGSOC() {
        BackgroundSubtractorGSOC bgs = Bgsegm.createBackgroundSubtractorGSOC();
        assertNotNull("could not create a GSOC instance!", bgs);
        Mat mask = new Mat();
        bgs.apply(img,mask);
        assertFalse("no mask created from GSOC", mask.empty());
    }
}
