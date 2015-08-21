using UnityEditor;
using UnityEngine;
using System.Collections.Generic;

[CustomEditor(typeof(HomographyPlane))]
public class HomographyEditor : Editor
{
    private void OnSceneGUI()
    {
        HomographyPlane hPlane = target as HomographyPlane;

        var hPlaneGroup = hPlane.transform.parent.GetComponentsInChildren<HomographyPlane>();
        var neighborCorners = new List<Vector3>();
        foreach(var hp in hPlaneGroup)
        {
            if(hp == hPlane)
            {
                continue;
            }
            Transform hTransform = hp.transform;
            for (int i = 0; i < hp.p.Length; i++)
            {
                neighborCorners.Add(hTransform.TransformPoint(hp.p[i]));
            }
        }

        Transform handleTransform = hPlane.transform;
        Quaternion handleRotation = Tools.pivotRotation == PivotRotation.Local ?
                    handleTransform.rotation : Quaternion.identity;
        Vector3[] p = new Vector3[4];
        for(int i = 0; i < p.Length; i++)
        {
            p[i] = handleTransform.TransformPoint(hPlane.p[i]);
        }

        Handles.color = Color.white;
        for (int i = 0; i < p.Length; i++)
        {
            Handles.DrawLine(p[i], p[(i + 1) % 4]);
            Handles.DoPositionHandle(p[i], handleRotation);
        }

        for (int i = 0; i < p.Length; i++)
        {
            EditorGUI.BeginChangeCheck();
            p[i] = Handles.DoPositionHandle(p[i], handleRotation);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(hPlane, "Move Point");
                EditorUtility.SetDirty(hPlane);

                // snapping
                float closestDistance = 1e5f;
                Vector3 closestCorner = Vector3.zero;
                foreach (var corner in neighborCorners)
                {
                    float distance = (corner - p[i]).sqrMagnitude;
                    if(closestDistance > distance)
                    {
                        closestDistance = distance;
                        closestCorner = corner;
                    }
                }
                if(closestDistance < 50)
                {
                    p[i] = closestCorner;
                }
                hPlane.p[i] = handleTransform.InverseTransformPoint(p[i]);
            }
        }
    }
}