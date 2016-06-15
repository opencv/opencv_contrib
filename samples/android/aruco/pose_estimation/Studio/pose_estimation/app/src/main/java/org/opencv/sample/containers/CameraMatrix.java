package org.opencv.sample.containers;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Base64;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.opencv.core.Mat;
import org.opencv.sample.utilities.CONSTANTS;
import org.opencv.sample.utilities.DEFAULT_PARAMS;

/**
 * Created by Sarthak on 04/06/16.
 */
public class CameraMatrix {
    SharedPreferences sharedPreferences;
    SharedPreferences.Editor editor;

    public static Mat MainMatrix;
    public static Mat DistortionCoefficients;

    public static long MainMatrix_ADDRESS;
    public static long DistortionCoefficients_ADDRESS;

    Context context;

    public CameraMatrix(Context c) {
        context=c;
        sharedPreferences = c.getSharedPreferences(CONSTANTS.Params, c.MODE_PRIVATE);
        editor = sharedPreferences.edit();
        loadMatrix();
    }

    public void loadMatrix(){

        MainMatrix = matFromJson(sharedPreferences.getString(CONSTANTS.PREF_MAIN_MATRIX,DEFAULT_PARAMS.DEFAULT_MAIN_MATRIX));
        DistortionCoefficients = matFromJson(sharedPreferences.getString(CONSTANTS.PREF_DIST_COEFF, DEFAULT_PARAMS.DEFAULT_DIST_COEFF));
        MainMatrix_ADDRESS=MainMatrix.getNativeObjAddr();
        DistortionCoefficients_ADDRESS=DistortionCoefficients.getNativeObjAddr();

    }

    public static String matToJson(Mat mat){
        JsonObject obj = new JsonObject();

        if(mat.isContinuous()){
            int cols = mat.cols();
            int rows = mat.rows();
            int elemSize = (int) mat.elemSize();

            byte[] data = new byte[cols * rows * elemSize];

            mat.get(0, 0, data);

            obj.addProperty("rows", mat.rows());
            obj.addProperty("cols", mat.cols());
            obj.addProperty("type", mat.type());

            // We cannot set binary data to a json object, so:
            // Encoding data byte array to Base64.
            String dataString = new String(Base64.encode(data, Base64.DEFAULT));

            obj.addProperty("data", dataString);

            Gson gson = new Gson();
            String json = gson.toJson(obj);

            return json;
        } else {
            Log.e("OpenCV", "Mat not continuous.");
        }
        return "{}";
    }

    public static Mat matFromJson(String json){
        JsonParser parser = new JsonParser();
        JsonObject JsonObject = parser.parse(json).getAsJsonObject();
        System.out.println(json);
        int rows = JsonObject.get("rows").getAsInt();
        int cols = JsonObject.get("cols").getAsInt();
        int type = JsonObject.get("type").getAsInt();

        String dataString = JsonObject.get("data").getAsString();
        byte[] data = Base64.decode(dataString.getBytes(), Base64.DEFAULT);

        Mat mat = new Mat(rows, cols, type);
        mat.put(0, 0, data);

        return mat;
    }

    public void saveMatrix(){
        editor.putString(CONSTANTS.PREF_MAIN_MATRIX,matToJson(MainMatrix));
        editor.putString(CONSTANTS.PREF_DIST_COEFF, matToJson(DistortionCoefficients));
        editor.commit();
    }
    public void resetMatrix(){
        editor.remove(CONSTANTS.PREF_MAIN_MATRIX);
        editor.remove(CONSTANTS.PREF_DIST_COEFF);
        MainMatrix=null;
        DistortionCoefficients=null;
        MainMatrix_ADDRESS=0;
        DistortionCoefficients_ADDRESS=0;
    }
}
