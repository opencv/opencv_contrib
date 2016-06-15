package org.opencv.sample.app;

/**
 * Created by Sarthak on 29/05/16.
 */

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
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.sample.poseestimation.R;


public class EstimationActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::EstActivity";

    private Mat                    mGray;

    private BroadcastReceiver receiver;
    private CameraBridgeViewBase   mOpenCvCameraView;
    private ScrollView mLogScrollView;
    private TextView mLog;
    private Button mButtonStart;
    private Button mButtonStop;

    private RadioButton mRadioButtonArucoB;
    private RadioButton mRadioButtonCharucoB;
    private RadioButton mRadioButtonCharucoD;

    private Boolean running=false;

    private enum DETECTION_ENTITY{
        ARUCO_BOARD, CHARUCO_BOARD, CHARUCO_DIAMOND
    }

    private DETECTION_ENTITY detection_entity=DETECTION_ENTITY.CHARUCO_DIAMOND;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
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

        setContentView(R.layout.activity_estimation);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mLogScrollView = (ScrollView) findViewById(R.id.scrollView_log_container);
        mLog = (TextView) findViewById(R.id.textView_log_content);
        mButtonStart= (Button)findViewById(R.id.button_start);
        mButtonStop=(Button) findViewById(R.id.button_stop);
        mRadioButtonArucoB = (RadioButton)findViewById(R.id.radioButton_aruco_b);
        mRadioButtonCharucoB = (RadioButton)findViewById(R.id.radioButton_charuco_b);
        mRadioButtonCharucoD = (RadioButton)findViewById(R.id.radioButton_charuco_d);
        IntentFilter filter = new IntentFilter();
        filter.addAction("ADDLOG_E");

        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                System.out.println(intent.getStringExtra("log"));
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
                    detection_entity=DETECTION_ENTITY.CHARUCO_BOARD;
                }
            }
        });

        mRadioButtonCharucoD.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (((RadioButton)v).isChecked()){
                    detection_entity=DETECTION_ENTITY.CHARUCO_DIAMOND;
                }
            }
        });

        mRadioButtonArucoB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (((RadioButton)v).isChecked()){
                    detection_entity=DETECTION_ENTITY.ARUCO_BOARD;
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
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        Intent intent = new Intent("ADDLOG_E");
        switch (detection_entity){
            case ARUCO_BOARD:
                intent.putExtra("log",NativeClass.EstimatePoseArucoBoard(mGray.getNativeObjAddr(),MainActivity.parameters,MainActivity.cameraMatrix));
                break;
            case CHARUCO_BOARD:
                intent.putExtra("log",NativeClass.EstimatePoseCharucoBoard(mGray.getNativeObjAddr(),MainActivity.parameters,MainActivity.cameraMatrix));
                break;
            case CHARUCO_DIAMOND:
                intent.putExtra("log",NativeClass.EstimatePoseCharucoDiamond(mGray.getNativeObjAddr(),MainActivity.parameters,MainActivity.cameraMatrix));
        }
        sendBroadcast(intent);
        return mGray;
    }

    public void setButtons(){
        if (running==true){
            mButtonStart.setEnabled(false);
            mButtonStop.setEnabled(true);
        }
        else{
            mButtonStart.setEnabled(true);
            mButtonStop.setEnabled(false);
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
