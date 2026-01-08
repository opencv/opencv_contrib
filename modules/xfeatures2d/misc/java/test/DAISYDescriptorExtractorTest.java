package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.xfeatures2d.DAISY;

public class DAISYDescriptorExtractorTest extends OpenCVTestCase {

    DAISY extractor;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = DAISY.create(); // default (15, 3, 8, 8, 100, noArray, true, false)
    }

    public void testCreate() {
        assertNotNull(extractor);
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPointMat() {
        fail("Not yet implemented");
    }

    public void testEmpty() {
        fail("Not yet implemented");
    }

    public void testReadYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.DAISY\"\nradius: 16.\nq_radius: 4\nq_theta: 9\nq_hist: 10\nnorm_type: 101\nenable_interpolation: 0\nuse_orientation: 1\n");

        extractor.read(filename);

        assertEquals(16.0f, extractor.getRadius());
        assertEquals(4, extractor.getQRadius());
        assertEquals(9, extractor.getQTheta());
        assertEquals(10, extractor.getQHist());
        assertEquals(101, extractor.getNorm());
        assertEquals(false, extractor.getInterpolation());
        assertEquals(true, extractor.getUseOrientation());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.DAISY\"\nradius: 15.\nq_radius: 3\nq_theta: 8\nq_hist: 8\nnorm_type: 100\nenable_interpolation: 1\nuse_orientation: 0\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
