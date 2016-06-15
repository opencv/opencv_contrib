package org.opencv.sample.app;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.sample.poseestimation.R;
import org.opencv.sample.utilities.CONSTANTS;
import org.opencv.sample.utilities.DEFAULT_PARAMS;

public class SettingsActivity extends AppCompatActivity {

    /**
     * The {@link android.support.v4.view.PagerAdapter} that will provide
     * fragments for each of the sections. We use a
     * {@link FragmentPagerAdapter} derivative, which will keep every
     * loaded fragment in memory. If this becomes too memory intensive, it
     * may be best to switch to a
     * {@link android.support.v4.app.FragmentStatePagerAdapter}.
     */

    private SectionsPagerAdapter mSectionsPagerAdapter;

    /**
     * The {@link ViewPager} that will host the section contents.
     */
    private ViewPager mViewPager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        final SharedPreferences sharedPreferences=getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        // Create the adapter that will return a fragment for each of the three
        // primary sections of the activity.
        mSectionsPagerAdapter = new SectionsPagerAdapter(getSupportFragmentManager());


        // Set up the ViewPager with the sections adapter.
        mViewPager = (ViewPager) findViewById(R.id.container);
        mViewPager.setAdapter(mSectionsPagerAdapter);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(SettingsActivity.this, "Changes will be saved on exit. In case of blank input defaults will be plugged in.", Toast.LENGTH_SHORT).show();
            }
        });

    }

    /**
     * Main Parameter fragment containing the Main Params.
     */
/*    public static class MainParamsFragment extends Fragment {
        public MainParamsFragment() {
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = inflater.inflate(R.layout.fragment_mainparams, container, false);
            getActivity().setTitle("MAIN PARAMS");
            TextView textView = (TextView) rootView.findViewById(R.id.section_label);
            textView.setText("Not Available. Edit Parameters in /mnt/sdcard/Pose_Estimation");
            return rootView;
        }
    }

    *//**
     * Main Parameter fragment containing the Main Params.
     *//*
    public static class CameraMatrixFragment extends Fragment {
        public CameraMatrixFragment() {
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = inflater.inflate(R.layout.fragment_cameramatrix, container, false);
            getActivity().setTitle("CAMERA MATRIX");
            TextView textView = (TextView) rootView.findViewById(R.id.section_label);
            textView.setText("Not Available. Edit Parameters in /mnt/sdcard/Pose_Estimation");
            return rootView;
        }
    }*/

    /**
     * Aruco Marker Parameter fragment
     */
    public static class ArucoMarkerFragment extends Fragment {
        /**
         * The fragment argument representing the section number for this
         * fragment.
         */

    }

    /**
     * ArucoBoard fragment.
     */
    public static class ArucoBoardFragment extends Fragment {
        public ArucoBoardFragment() {
        }
        EditText arucob_s;
        EditText arucob_l;
        EditText arucob_d;

        EditText arucob_w;
        EditText arucob_h;

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = inflater.inflate(R.layout.fragment_arucoboardparams, container, false);

            TextView textView = (TextView) rootView.findViewById(R.id.section_label);
            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            arucob_d=(EditText)rootView.findViewById(R.id.arucoboard_d);
            arucob_l=(EditText)rootView.findViewById(R.id.arucoboard_l);
            arucob_s=(EditText)rootView.findViewById(R.id.arucoboard_s);
            arucob_w=(EditText)rootView.findViewById(R.id.arucoboard_w);
            arucob_h=(EditText)rootView.findViewById(R.id.arucoboard_h);
            arucob_d.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_ARUCO_BOARD_D,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_D)));
            arucob_l.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_ARUCO_BOARD_L,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_L)));
            arucob_s.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_ARUCO_BOARD_S,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_S)));
            arucob_w.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_ARUCO_BOARD_W,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_W)));
            arucob_h.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_ARUCO_BOARD_H,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_H)));
            return rootView;
        }

        @Override
        public void onStop() {
            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            if(arucob_d.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_D,Integer.parseInt(arucob_d.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_D,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_D).commit();
            }
            if(arucob_l.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_ARUCO_BOARD_L,Float.parseFloat(arucob_l.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_ARUCO_BOARD_L,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_L).commit();
            }
            if(arucob_s.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_ARUCO_BOARD_S,Float.parseFloat(arucob_s.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_ARUCO_BOARD_S,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_S).commit();
            }
            if(arucob_w.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_W,Integer.parseInt(arucob_w.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_W,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_W).commit();
            }
            if(arucob_h.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_H,Integer.parseInt(arucob_h.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_ARUCO_BOARD_H,DEFAULT_PARAMS.DEFAULT_ARUCO_BOARD_H).commit();
            }
            MainActivity.parameters.loadParameters();
            Toast.makeText(getActivity(), "Updated", Toast.LENGTH_SHORT).show();
            super.onStop();
        }
    }

    /**
     * CharucoBoard fragment
     */
    public static class CharucoBoardFragment extends Fragment {
        /**
         * The fragment argument representing the section number for this
         * fragment.
         */
        EditText charucob_sl;
        EditText charucob_ml;
        EditText charucob_d;
        EditText charucob_w;
        EditText charucob_h;


        public CharucoBoardFragment() {
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = inflater.inflate(R.layout.fragment_charucoboardparams, container, false);

            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            charucob_d=(EditText)rootView.findViewById(R.id.charucoboard_d);
            charucob_ml=(EditText)rootView.findViewById(R.id.charucoboard_ml);
            charucob_sl=(EditText)rootView.findViewById(R.id.charucoboard_sl);
            charucob_w=(EditText)rootView.findViewById(R.id.charucoboard_w);
            charucob_h=(EditText)rootView.findViewById(R.id.charucoboard_h);
            charucob_d.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_CHARUCO_BOARD_D,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_D)));
            charucob_ml.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_CHARUCO_BOARD_ML,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_ML)));
            charucob_sl.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_CHARUCO_BOARD_SL,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_SL)));
            charucob_h.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_CHARUCO_BOARD_H,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_H)));
            charucob_w.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_CHARUCO_BOARD_W,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_W)));
            TextView textView = (TextView) rootView.findViewById(R.id.section_label);
            return rootView;
        }

        @Override
        public void onStop() {
            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            if(charucob_d.getText().toString().trim().length()!=0){

                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_D,Integer.parseInt(charucob_d.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_D,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_D).commit();
            }
            if(charucob_ml.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_BOARD_ML,Float.parseFloat(charucob_ml.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_BOARD_ML,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_ML).commit();
            }
            if(charucob_sl.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_BOARD_SL,Float.parseFloat(charucob_sl.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_BOARD_SL,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_SL).commit();
            }
            if(charucob_w.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_W,Integer.parseInt(charucob_w.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_W,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_W).commit();
            }
            if(charucob_h.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_H,Integer.parseInt(charucob_h.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_BOARD_H,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_H).commit();
            }
            MainActivity.parameters.loadParameters();
            Toast.makeText(getActivity(), "Updated", Toast.LENGTH_SHORT).show();
            super.onStop();
        }
    }

    /**
     * CharucoDiamond fragment containing the Main Params.
     */
    public static class CharucoDiamondFragment extends Fragment {
        /**
         * The fragment argument representing the section number for this
         * fragment.
         */
        EditText charucod_sl;
        EditText charucod_ml;
        EditText charucod_d;

        public CharucoDiamondFragment() {

        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View rootView = inflater.inflate(R.layout.fragment_charucodiamondparams, container, false);

            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            TextView textView = (TextView) rootView.findViewById(R.id.section_label);
            charucod_sl=(EditText)rootView.findViewById(R.id.charucodiamond_sl);
            charucod_ml=(EditText)rootView.findViewById(R.id.charucodiamond_ml);
            charucod_d=(EditText)rootView.findViewById(R.id.charucodiamond_d);
            charucod_d.setText(String.valueOf(sharedPreferences.getInt(CONSTANTS.PREF_CHARUCO_DIAMOND_D, DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_D)));
            charucod_ml.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_ML,DEFAULT_PARAMS.DEFAULT_CHARUCO_BOARD_ML)));
            charucod_sl.setText(String.valueOf(sharedPreferences.getFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_SL,DEFAULT_PARAMS.DEFAULT_CHARUCO_DIAMOND_SL)));
            return rootView;
        }

        @Override
        public void onStop() {
            SharedPreferences sharedPreferences=getActivity().getApplicationContext().getSharedPreferences(CONSTANTS.Params, MODE_PRIVATE);
            if(charucod_d.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_DIAMOND_D,Integer.parseInt(charucod_d.getText().toString())).commit();

            }else{
                sharedPreferences.edit().putInt(CONSTANTS.PREF_CHARUCO_DIAMOND_D,DEFAULT_PARAMS.DEFAULT_CHARUCO_DIAMOND_D).commit();
            }
            if(charucod_ml.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_ML,Float.parseFloat(charucod_ml.getText().toString())).apply();

            }else{
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_ML,DEFAULT_PARAMS.DEFAULT_CHARUCO_DIAMOND_ML).apply();
            }
            if(charucod_sl.getText().toString().trim().length()!=0){
                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_SL,Float.parseFloat(charucod_sl.getText().toString())).apply();

            }else{

                sharedPreferences.edit().putFloat(CONSTANTS.PREF_CHARUCO_DIAMOND_SL,DEFAULT_PARAMS.DEFAULT_CHARUCO_DIAMOND_SL).apply();
            }
            MainActivity.parameters.loadParameters();
            Toast.makeText(getActivity(), "Updated", Toast.LENGTH_SHORT).show();
            super.onStop();
        }
    }


    /**
     * A {@link FragmentPagerAdapter} that returns a fragment corresponding to
     * one of the sections/tabs/pages.
     */
    public class SectionsPagerAdapter extends FragmentPagerAdapter {

        public SectionsPagerAdapter(FragmentManager fm) {
            super(fm);
        }

        @Override
        public Fragment getItem(int position) {
            // getItem is called to instantiate the fragment for the given page.
            // Return a PlaceholderFragment (defined as a static inner class below).
            switch (position){
                case 0: return new ArucoBoardFragment();
                case 1: return new CharucoBoardFragment();
                case 2: return new CharucoDiamondFragment();
//                case 4: return new MainParamsFragment();
//                case 5: return new CameraMatrixFragment();
                default: return null;
            }
        }

        @Override
        public int getCount() {
            // Show 3 total pages.
            return 3;
        }
    }
}
