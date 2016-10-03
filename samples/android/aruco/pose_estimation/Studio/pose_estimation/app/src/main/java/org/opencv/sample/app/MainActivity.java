package org.opencv.sample.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.sample.containers.CameraMatrix;
import org.opencv.sample.containers.FileManager;
import org.opencv.sample.containers.Parameters;
import org.opencv.sample.poseestimation.R;

import java.io.IOException;

/**
 * Created by Sarthak on 29/05/16.
 */
public class MainActivity extends Activity {
    /** Called when the activity is first created. */
    private static final String    TAG                 = "OCVSample::MainActivity";

    /** UI*/
    private Button button_estimation;
    private Button button_calibration;
    private Button button_settings;

    /** Parameters*/
    public static Parameters parameters;
    public static CameraMatrix cameraMatrix;

    /**File Manager*/
    public static FileManager fileManager;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");

        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        button_estimation=(Button)findViewById(R.id.button_estimate);
        button_calibration=(Button)findViewById(R.id.button_calibrate);
        button_settings=(Button)findViewById(R.id.button_settings);

        parameters=new Parameters(getApplicationContext());
        parameters.resetParameters();
        //cameraMatrix = new CameraMatrix(getApplicationContext());

        System.out.println("CHANGE");

        try {
            fileManager.prepareSystem(this);
        } catch (IOException e) {
            e.printStackTrace();
        }

        button_estimation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent myIntent = new Intent(MainActivity.this, EstimationActivity.class);
                MainActivity.this.startActivity(myIntent);
            }
        });

        button_calibration.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent myIntent = new Intent(MainActivity.this, CalibrationActivity.class);
                MainActivity.this.startActivity(myIntent);
            }
        });

        button_settings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent myIntent = new Intent(MainActivity.this, SettingsActivity.class);
                MainActivity.this.startActivity(myIntent);
            }
        });
    }
}
