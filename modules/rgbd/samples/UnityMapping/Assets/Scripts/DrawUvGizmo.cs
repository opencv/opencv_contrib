using UnityEngine;
using System.Collections;

public class DrawUvGizmo : MonoBehaviour
{
    public GameObject targetObject;
    public Mesh textureMesh;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.green;

/*        for (int i = 0; i < mesh.vertexCount; i++)
        {
            Gizmos.DrawSphere(mesh.uv[i] * 100 + renderPosition, 1);
        }*/

/*        if(textureMesh == null)
        {
            var mesh = targetObject.GetComponent<MeshFilter>().sharedMesh;
            textureMesh = new Mesh();
            
            textureMesh.vertices = new Vector3[mesh.vertexCount];
            textureMesh.normals = new Vector3[mesh.vertexCount];
            for (int i = 0; i < mesh.vertexCount; i++)
            {
                textureMesh.vertices[i] = mesh.uv[i] * 100;
                textureMesh.normals[i] = new Vector3(0, 0, 1);
            }
            textureMesh.SetIndices(mesh.GetIndices(0), MeshTopology.Triangles, 0);
        }*/
        Gizmos.DrawWireMesh(targetObject.transform.GetChild(0).gameObject.GetComponent<MeshFilter>().sharedMesh, 0, transform.position + new Vector3(-50, -50, 0));
    }

    
}
