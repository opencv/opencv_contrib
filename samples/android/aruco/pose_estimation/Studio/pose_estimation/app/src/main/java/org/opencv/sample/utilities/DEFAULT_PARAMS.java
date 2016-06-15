package org.opencv.sample.utilities;

/**
 * Created by Sarthak on 29/05/16.
 */

public class DEFAULT_PARAMS {

    public static final String DEFAULT_MAIN_MATRIX= "{\"rows\": \"3\",\"cols\": \"3\", \"type\": \"0\", \"data\":\"Y29yZV9NYXRf\\n\"}";
    public static final String DEFAULT_DIST_COEFF= "{\"rows\": \"1\",\"cols\": \"5\", \"type\": \"0\", \"data\":\"Y29yZV9NYXRf\\n\"}";

    //GENERAL
    public static final int DEFAULT_NMARKERS= 1024;
    public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_MIN= 3;
    public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_MAX= 23;
    public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_STEP= 10;
    public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE= 21;
    public static final float DEFAULT_ADAPTIVE_THRESH_CONSTANT= (float) 7.0;
    public static final float DEFAULT_MIN_MARKER_PERIMETER_RATE= (float) 0.03;
    public static final float DEFAULT_MAX_MARKER_PERIMETER_RATE= (float) 4.0;
    public static final float DEFAULT_POLYGONAL_APPROX_ACCURACY_RATE= (float)  0.05;
    public static final float DEFAULT_MIN_CORNER_DISTANCE= (float)  10.0;
    public static final int DEFAULT_MIN_DISTANCE_TO_BORDER= 3;
    public static final float DEFAULT_MIN_MARKER_DISTANCE= (float)  10.0;
    public static final float DEFAULT_MIN_MARKER_DISTANCE_RATE= (float)  0.05;
    public static final boolean DEFAULT_DO_CORNER_REFINEMENT= true;
    public static final int DEFAULT_CORNER_REFINEMENT_WIN_SIZE= 5;
    public static final int DEFAULT_CORNER_REFINEMENT_MAX_ITERATIONS= 30;
    public static final float DEFAULT_CORNER_REFINEMENT_MIN_ACCURACY= (float)  0.1;
    public static final int DEFAULT_MARKER_BORDER_BITS= 1;
    public static final int  DEFAULT_PERSPECTIVE_REMOVE_PIXEL_PER_CELL= 8;
    public static final float DEFAULT_PERSEPCTIVE_REMOVE_IGNORE_MARGIN_PER_CELL= (float)  0.13;
    public static final float DEFAULT_MAX_ERRONEOUS_BITS_IN_BORDER_RATE= (float)  0.04;
    public static final float DEFAULT_MIN_OTSU_STD_DEV= (float)  5.0;
    public static final float DEFAULT_ERROR_CORRECTION_RATE= (float)  0.6;


    //ARUCO MARKER
    public static final int DEFAULT_ARUCO_MARKER_D= 10;

    //ARUCO BOARD
    public static final int DEFAULT_ARUCO_BOARD_D= 10;
    public static final int DEFAULT_ARUCO_BOARD_W= 5;
    public static final int DEFAULT_ARUCO_BOARD_H= 7;
    public static final float DEFAULT_ARUCO_BOARD_S= (float)10.0;
    public static final float DEFAULT_ARUCO_BOARD_L= (float)100.0;

    //CHARUCO BOARD
    public static final int DEFAULT_CHARUCO_BOARD_D= 10;
    public static final int DEFAULT_CHARUCO_BOARD_W= 5;
    public static final int DEFAULT_CHARUCO_BOARD_H= 7;
    public static final float DEFAULT_CHARUCO_BOARD_SL= (float)  0.04;
    public static final float DEFAULT_CHARUCO_BOARD_ML= (float)  0.02;

    //CHARUCO CHARUCO DIAMOND
    public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
    public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
    public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
}
