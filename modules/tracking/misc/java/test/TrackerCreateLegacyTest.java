package org.opencv.test.tracking;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.test.OpenCVTestCase;

import org.opencv.tracking.Tracking;
import org.opencv.tracking.legacy_Tracker;
import org.opencv.tracking.legacy_TrackerTLD;
import org.opencv.tracking.legacy_MultiTracker;

public class TrackerCreateLegacyTest extends OpenCVTestCase {

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }


    public void testCreateLegacyTrackerTLD() {
        legacy_Tracker tracker = legacy_TrackerTLD.create();
    }

    public void testCreateLegacyMultiTracker() {
        legacy_MultiTracker multiTracker = legacy_MultiTracker.create();
        assert(multiTracker != null);
    }

    public void testAddLegacyMultiTracker() {
        legacy_MultiTracker multiTracker = legacy_MultiTracker.create();
        legacy_Tracker tracker = legacy_TrackerTLD.create();
        Mat image = new Mat(100, 100, CvType.CV_8UC3);
        Rect2d boundingBox = new Rect2d(10, 10, 50, 50);

        boolean result = multiTracker.add(tracker, image, boundingBox);
        assert(result);
    }

}
