using UnityEngine;
using System.Collections;

public class ProjectionManager : MonoBehaviour {

    public int projectorWidth = 1024;
    public int projectorHeight = 768;
    Camera cam;
    public GameObject projectorImage;

    ProjectionManager()
    {
        projectorWidth = Screen.currentResolution.width;
        projectorHeight = Screen.currentResolution.height;
    }

    // Use this for initialization
    void Start () {
        UpdateView();
    }

    // Update is called once per frame
    void Update () {
	
	}

    void OnDrawGizmos()
    {
        UpdateView();
    }

    void UpdateView()
    {
        transform.position = new Vector3(projectorWidth / 2, -projectorHeight / 2, -10);

        cam = gameObject.GetComponent<Camera>();
        cam.orthographicSize = projectorHeight / 2;

        projectorImage.transform.localPosition = new Vector3(0, 0, 5000);
        projectorImage.transform.localScale = new Vector3(100, 100);
    }
}
