package org.opencv.test.aruco;

import java.util.ArrayList;
import java.util.List;

import org.opencv.test.OpenCVTestCase;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.aruco.*;


public class ArucoTest extends OpenCVTestCase {

    public void testArucoIssue3133() {
        byte[][] marker = {{0,1,1},{1,1,1},{0,1,1}};
        Dictionary dictionary = Dictionary.create(1, 3);
        dictionary.set_maxCorrectionBits(0);
        Mat markerBits = new Mat(3, 3, CvType.CV_8UC1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                markerBits.put(i, j, marker[i][j]);
            }
        }

        Mat markerCompressed = Dictionary.getByteListFromBits(markerBits);
        assertMatNotEqual(markerCompressed, dictionary.get_bytesList());

        dictionary.set_bytesList(markerCompressed);
        assertMatEqual(markerCompressed, dictionary.get_bytesList());
    }

    public void testArucoDetector() {
        Dictionary dictionary = Dictionary.get(0);
        DetectorParameters detectorParameters = DetectorParameters.create();
        ArucoDetector detector = ArucoDetector.create(dictionary, detectorParameters);

        Mat markerImage = new Mat();
        int id = 1, offset = 5, size = 40;
        Aruco.drawMarker(dictionary, id, size, markerImage, detectorParameters.get_markerBorderBits());

        Mat image = new Mat(markerImage.rows() + 2*offset, markerImage.cols() + 2*offset,
                            CvType.CV_8UC1, new Scalar(255));
        Mat m = image.submat(offset, size+offset, offset, size+offset);
        markerImage.copyTo(m);

        List<Mat> corners = new ArrayList();
        Mat ids = new Mat();
        detector.detectMarkers(image, corners, ids);

        assertEquals(1, corners.size());
        Mat res = corners.get(0);
        assertArrayEquals(new double[]{offset, offset}, res.get(0, 0), 0.0);
        assertArrayEquals(new double[]{size + offset - 1, offset}, res.get(0, 1), 0.0);
        assertArrayEquals(new double[]{size + offset - 1, size + offset - 1}, res.get(0, 2), 0.0);
        assertArrayEquals(new double[]{offset, size + offset - 1}, res.get(0, 3), 0.0);
    }
}
