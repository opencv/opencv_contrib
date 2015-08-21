using UnityEngine;
using UnityEditor;
using System.Diagnostics;

public class Calibration : EditorWindow
{
    int deviceID = 0;
    int lightThreshold = 10;
    int lightIntensity = 50;
    int projectorWidth = 1024;
    int projectorHeight = 768;
    bool useKinect = false;

    string exePath;
    string workingPath;

    Calibration()
    {
    }

    // Add menu named "My Window" to the Window menu
    [MenuItem("Window/Calibration")]
    static void Init()
    {
        // Get existing open window or if none, make a new one:
        Calibration window = (Calibration)EditorWindow.GetWindow(typeof(Calibration));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Calibration Manager", EditorStyles.boldLabel);

        if (GUILayout.Button("Select Folder Containing Executables"))
        {
            exePath = EditorUtility.OpenFolderPanel(
                "Select folder containing executables",
                "",
                "");
        }
        exePath = EditorGUILayout.TextField("Executables", exePath);

        if (GUILayout.Button("Select Assets Folder"))
        {
            workingPath = EditorUtility.OpenFolderPanel(
                "Select assets folder",
                "",
                "");
        }
        workingPath = EditorGUILayout.TextField("Assets Folder", workingPath);

        GUILayout.Label("Calibration Settings", EditorStyles.boldLabel);

        deviceID = EditorGUILayout.IntField("Device ID", deviceID);
        lightThreshold = EditorGUILayout.IntField("Light Threshold", lightThreshold);
        lightIntensity = EditorGUILayout.IntField("Light Intensity", lightIntensity);
        projectorWidth = EditorGUILayout.IntField("Projector Width", projectorWidth);
        projectorHeight = EditorGUILayout.IntField("Projector Height", projectorHeight);
        useKinect = EditorGUILayout.Toggle("Use Kinect", useKinect);

        int useKinectInt;
        if (useKinect)
        {
            useKinectInt = 1;
        }
        else
        {
            useKinectInt = 0;
        }

        if(GUILayout.Button("Start Scanning"))
        {
            Process process = new Process();
            process.StartInfo.FileName = exePath + "\\rgbd-example-projection_calibration.exe";
            process.StartInfo.WorkingDirectory = workingPath + "\\Opencv";
            process.StartInfo.Arguments = "-threshold=" + lightThreshold.ToString() + " -intensity=" + lightIntensity.ToString() + " -w=" + projectorWidth.ToString() + " -h=" + projectorHeight.ToString() + " -openni=" + useKinectInt.ToString();
            process.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
            process.Start();
        }

        if (GUILayout.Button("Start Meshing"))
        {
            Process process = new Process();
            process.StartInfo.FileName = exePath + "\\rgbd-example-projection_meshing.exe";
            process.StartInfo.WorkingDirectory = workingPath + "\\Opencv";
            process.StartInfo.Arguments = "-w=" + projectorWidth.ToString() + " -h=" + projectorHeight.ToString() + " -openni=" + useKinectInt.ToString();
            process.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
            process.Start();
        }
    }



}