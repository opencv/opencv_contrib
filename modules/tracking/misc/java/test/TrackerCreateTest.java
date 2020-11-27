package org.opencv.test.tracking;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.test.OpenCVTestCase;

import org.opencv.video.Tracker;
import org.opencv.video.TrackerGOTURN;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.TrackerMIL;

public class TrackerCreateTest extends OpenCVTestCase {

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }


    public void testCreateTrackerGOTURN() {
        try {
            Tracker tracker = TrackerGOTURN.create();
            assert(tracker != null);
        } catch (CvException e) {
            // expected, model files may be missing
        }
    }

    public void testCreateTrackerKCF() {
        Tracker tracker = TrackerKCF.create();
    }

    public void testCreateTrackerMIL() {
        Tracker tracker = TrackerMIL.create();
    }

}
