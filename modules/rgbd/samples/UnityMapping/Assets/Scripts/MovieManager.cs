using UnityEngine;
using System.Collections;

public class MovieManager : MonoBehaviour {

	// Use this for initialization
	void Start () {
		var movieTexture = ((MovieTexture)GetComponent<Renderer> ().material.mainTexture);
		movieTexture.loop = true;
		movieTexture.Play();
		movieTexture.wrapMode = TextureWrapMode.Repeat;
	}
	
	// Update is called once per frame
	void Update () {

	}
}
