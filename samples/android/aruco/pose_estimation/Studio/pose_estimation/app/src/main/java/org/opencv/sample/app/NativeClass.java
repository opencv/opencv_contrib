package org.opencv.sample.app;

import org.opencv.sample.containers.CameraMatrix;
import org.opencv.sample.containers.Parameters;

/**
 * Created by Sarthak on 29/05/16.
 */

public class NativeClass {
    public native static String EstimatePoseArucoMarker(long inputImage, Parameters parameters, CameraMatrix cameraMatrix);
    public native static String EstimatePoseArucoBoard(long inputImage, Parameters parameters, CameraMatrix cameraMatrix);
    public native static String EstimatePoseCharucoBoard(long inputImage, Parameters parameters, CameraMatrix cameraMatrix);
    public native static String EstimatePoseCharucoDiamond(long inputImage, Parameters parameters, CameraMatrix cameraMatrix);
    public native static String CalibrateCameraCharucoBoard(Parameters parameters, CameraMatrix cameraMatrix);
    public native static String CalibrateCameraArucoBoard(Parameters parameters, CameraMatrix cameraMatrix);

    public native static String SendCalibrationImage(long inputImage);
    public native static String ResetCalibration();
}