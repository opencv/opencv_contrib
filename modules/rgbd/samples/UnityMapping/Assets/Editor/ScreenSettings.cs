using UnityEngine;
using UnityEditor;
public class ScreenSettings : EditorWindow
{
    // http://www.reddit.com/r/Unity3D/comments/2lymim/full_full_screen_on_play_script_freebie_for/

    bool doFullscreen = false;
    bool doFullscreenOld = false;

    int primaryScreenWidth;
    int secondaryScreenWidth;
    int secondaryScreenHeight;
    int primaryScreenWidthOld;
    int secondaryScreenWidthOld;
    int secondaryScreenHeightOld;

    ScreenSettings()
    {
        primaryScreenWidth = Screen.currentResolution.width;
        secondaryScreenWidth = Screen.currentResolution.width;
        secondaryScreenHeight = Screen.currentResolution.height;
        primaryScreenWidthOld = Screen.currentResolution.width;
        secondaryScreenWidthOld = Screen.currentResolution.width;
        secondaryScreenHeightOld = Screen.currentResolution.height;
    }

    private static int tabHeight = 22;

    // Add menu named "My Window" to the Window menu
    [MenuItem("Window/Screen Settings")]
    static void Init()
    {
        // Get existing open window or if none, make a new one:
        ScreenSettings window = (ScreenSettings)EditorWindow.GetWindow(typeof(ScreenSettings));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Screen Settings", EditorStyles.boldLabel);
        GUILayout.Label("Keep this window floating!", EditorStyles.label);
        primaryScreenWidth = EditorGUILayout.IntField("Monitor Width", primaryScreenWidth);
        secondaryScreenWidth = EditorGUILayout.IntField("Projector Width", secondaryScreenWidth);
        secondaryScreenHeight = EditorGUILayout.IntField("Projector Height", secondaryScreenHeight);
        doFullscreen = EditorGUILayout.Toggle("Go Fullscreen", doFullscreen);
    }

    void Update()
    {
        if (doFullscreen != doFullscreenOld // toggle is changed
            || !( // or, values are changed
             primaryScreenWidth == primaryScreenWidthOld
             && secondaryScreenWidth == secondaryScreenWidthOld
             && secondaryScreenHeight == secondaryScreenHeightOld
            )
            )
        {
            if (doFullscreen)
            {
                FullScreenGameWindow(primaryScreenWidth, secondaryScreenWidth, secondaryScreenHeight);
            }
            else if (doFullscreen != doFullscreenOld)
            {
                CloseGameWindow();
            }
            doFullscreenOld = doFullscreen;
            primaryScreenWidthOld = primaryScreenWidth;
            secondaryScreenWidthOld = secondaryScreenWidth;
            secondaryScreenHeightOld = secondaryScreenHeight;
            Focus();
        }

    }

    static EditorWindow GetMainGameView()
    {
        EditorApplication.ExecuteMenuItem("Window/Game");

        System.Type T = System.Type.GetType("UnityEditor.GameView,UnityEditor");
        System.Reflection.MethodInfo GetMainGameView = T.GetMethod("GetMainGameView", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        System.Object Res = GetMainGameView.Invoke(null, null);
        return (EditorWindow)Res;
    }

    public static void FullScreenGameWindow(int psw, int ssw, int ssh)
    {
        EditorWindow gameView = GetMainGameView();

        gameView.titleContent = new GUIContent("Game (Stereo)");
        Rect newPos = new Rect(psw, 0 - tabHeight, ssw, ssh + tabHeight);

        gameView.position = newPos;
        gameView.minSize = new Vector2(ssw, ssh + tabHeight);
        gameView.maxSize = gameView.minSize;
        gameView.position = newPos;
    }

    public static void CloseGameWindow()
    {
        EditorWindow gameView = GetMainGameView();
        gameView.Close();
    }
}