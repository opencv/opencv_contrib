package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.xfeatures2d.LUCID;

public class LUCIDDescriptorExtractorTest extends OpenCVTestCase {

    LUCID extractor;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = LUCID.create(); // default (1,2)
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
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.LUCID\"\nlucid_kernel: 2\nblur_kernel: 3\n");

        extractor.read(filename);

        assertEquals(2, extractor.getLucidKernel());
        assertEquals(3, extractor.getBlurKernel());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.LUCID\"\nlucid_kernel: 1\nblur_kernel: 2\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
