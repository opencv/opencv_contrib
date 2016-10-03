package org.opencv.sample.app;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.ScrollView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.sample.poseestimation.R;

/**
 * Created by Sarthak on 02/06/16.
 */
public class CalibrationActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::CalActivity";
    private BroadcastReceiver receiver;
    private Mat                    mBackup;
    private Mat                    mGray;

    private CameraBridgeViewBase mOpenCvCameraView;
    private ScrollView mLogScrollView;
    private TextView mLog;
    private Button mButtonStart;
    private Button mButtonStop;

    private Button mButtonClick;
    private Button mButtonCalibrate;

    private RadioButton mRadioButtonArucoB;
    private RadioButton mRadioButtonCharucoB;

    private Boolean running=false;

    private enum CALIBRATION_ENTITY{
        ARUCO_BOARD, CHARUCO_BOARD
    }

    private CALIBRATION_ENTITY detection_entity=CALIBRATION_ENTITY.CHARUCO_BOARD;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    addLog("OpenCV loaded successfully");
                    System.loadLibrary("opencv_mylib");
                    addLog("Ready");
                    setButtons();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_calibration);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mLogScrollView = (ScrollView) findViewById(R.id.scrollView_log_container);
        mLog = (TextView) findViewById(R.id.textView_log_content);
        mButtonStart= (Button)findViewById(R.id.button_start);
        mButtonStop=(Button) findViewById(R.id.button_stop);
        mButtonClick= (Button)findViewById(R.id.button_click);
        mButtonCalibrate=(Button) findViewById(R.id.button_calibrate);
        mRadioButtonArucoB = (RadioButton)findViewById(R.id.radioButton_aruco);
        mRadioButtonCharucoB = (RadioButton)findViewById(R.id.radioButton_charuco);

        IntentFilter filter = new IntentFilter();
        filter.addAction("ADDLOG_C");

        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                addLog(intent.getStringExtra("log"));
            }
        };
        registerReceiver(receiver, filter);

        mButtonStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clearLog();
                mOpenCvCameraView.enableView();
                running=true;
                setButtons();
            }
        });

        mButtonClick.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                addLog(NativeClass.SendCalibrationImage(mBackup.getNativeObjAddr()));
            }
        });

        mButtonCalibrate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent("ADDLOG_C");
                switch (detection_entity){
                    case ARUCO_BOARD:
                        intent.putExtra("log",NativeClass.CalibrateCameraArucoBoard(MainActivity.parameters,MainActivity.cameraMatrix));
                        break;
                    case CHARUCO_BOARD:
                        intent.putExtra("log",NativeClass.CalibrateCameraCharucoBoard(MainActivity.parameters,MainActivity.cameraMatrix));
                }
                sendBroadcast(intent);
                //try{
                //    MainActivity.cameraMatrix.saveMatrix();
                //}catch(Exception e){
                //    e.printStackTrace();
                //   addLog("Calibration could not be saved!");
                //}
            }
        });

        mButtonStop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mOpenCvCameraView.disableView();
                running=false;
                setButtons();
            }
        });

        mRadioButtonCharucoB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (((RadioButton)v).isChecked()){
                    detection_entity=CALIBRATION_ENTITY.CHARUCO_BOARD;
                }
            }
        });


        mRadioButtonArucoB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (((RadioButton)v).isChecked()){
                    detection_entity=CALIBRATION_ENTITY.ARUCO_BOARD;
                }
            }
        });

        setButtons();
        clearLog();
    }



    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            addLog("Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            addLog("OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        if (receiver != null) {
            unregisterReceiver(receiver);
            receiver = null;
        }
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        Intent intent = new Intent("ADDLOG_C");
        mBackup = mGray.clone();
        switch (detection_entity){
            case ARUCO_BOARD:
                intent.putExtra("log",NativeClass.EstimatePoseArucoBoard(mGray.getNativeObjAddr(),MainActivity.parameters,MainActivity.cameraMatrix));
                break;
            case CHARUCO_BOARD:
                intent.putExtra("log",NativeClass.EstimatePoseCharucoBoard(mGray.getNativeObjAddr(),MainActivity.parameters,MainActivity.cameraMatrix));
        }
        sendBroadcast(intent);
        return mGray;
    }

    public void setButtons(){
        if (running==true){
            mButtonStart.setEnabled(false);
            mButtonStop.setEnabled(true);
            mButtonCalibrate.setEnabled(false);
            mButtonClick.setEnabled(true);
        }
        else{
            mButtonStart.setEnabled(true);
            mButtonStop.setEnabled(false);
            mButtonClick.setEnabled(false);
            mButtonCalibrate.setEnabled(true);
        }
    }

    public void addLog(String log){
        Log.i(TAG, log);
        mLog.append(log+"\n");
        mLogScrollView.fullScroll(View.FOCUS_DOWN);
    }

    public void clearLog(){
        mLog.setText("");
    }
}
