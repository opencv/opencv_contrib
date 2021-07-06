package org.opencv.test.features2d;

import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.xfeatures2d.FREAK;

public class FREAKDescriptorExtractorTest extends OpenCVTestCase {

    FREAK extractor;

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        extractor = FREAK.create(); // default (true,true,22,4)
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
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.FREAK\"\norientationNormalized: 0\nscaleNormalized: 0\npatternScale: 23.\nnOctaves: 5\n");

        extractor.read(filename);

        assertEquals(false, extractor.getOrientationNormalized());
        assertEquals(false, extractor.getScaleNormalized());
        assertEquals(23.0, extractor.getPatternScale());
        assertEquals(5, extractor.getNOctaves());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        extractor.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.FREAK\"\norientationNormalized: 1\nscaleNormalized: 1\npatternScale: 22.\nnOctaves: 4\n";
        String actual = readFile(filename);
        actual = actual.replaceAll("e([+-])0(\\d\\d)", "e$1$2"); // NOTE: workaround for different platforms double representation
        assertEquals(truth, actual);
    }

}
