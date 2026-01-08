package org.opencv.test.tracking;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.test.OpenCVTestCase;
import org.opencv.dnn_superres.DnnSuperResImpl;

public class DnnSuperresTest extends OpenCVTestCase {

    public void testCreateSuperres() {
        DnnSuperResImpl sr = DnnSuperResImpl.create();
    }

}
