package org.opencv.test.tracking;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.test.OpenCVTestCase;

import org.opencv.tracking.Tracking;
import org.opencv.tracking.legacy_Tracker;
import org.opencv.tracking.legacy_TrackerTLD;

public class TrackerCreateLegacyTest extends OpenCVTestCase {

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }


    public void testCreateLegacyTrackerTLD() {
        legacy_Tracker tracker = legacy_TrackerTLD.create();
    }

}
