#include "precomp.hpp"
using namespace cv;
using namespace std;

namespace cv{ namespace cnn_3dobj{

	IcoSphere::IcoSphere(float radius_in, int depth_in)
	{

		X = 0.5f;
		Z = 0.5f;
		int radius = radius_in;
		int depth = depth_in;
		X *= radius;
		Z *= radius;

		float vdata[12][3] = { { -X, 0.0f, Z }, { X, 0.0f, Z },
				{ -X, 0.0f, -Z }, { X, 0.0f, -Z }, { 0.0f, Z, X }, { 0.0f, Z, -X },
				{ 0.0f, -Z, X }, { 0.0f, -Z, -X }, { Z, X, 0.0f }, { -Z, X, 0.0f },
				{ Z, -X, 0.0f }, { -Z, -X, 0.0f } };


		int tindices[20][3] = { { 0, 4, 1 }, { 0, 9, 4 }, { 9, 5, 4 },
				{ 4, 5, 8 }, { 4, 8, 1 }, { 8, 10, 1 }, { 8, 3, 10 }, { 5, 3, 8 },
				{ 5, 2, 3 }, { 2, 7, 3 }, { 7, 10, 3 }, { 7, 6, 10 }, { 7, 11, 6 },
				{ 11, 0, 6 }, { 0, 1, 6 }, { 6, 1, 10 }, { 9, 0, 11 },
				{ 9, 11, 2 }, { 9, 2, 5 }, { 7, 2, 11 } };

		std::vector<float>* texCoordsList = new std::vector<float>;
		std::vector<int>* indicesList = new std::vector<int>;

		// Iterate over points
		for (int i = 0; i < 20; ++i) {

			subdivide(vdata[tindices[i][1]], vdata[tindices[i][2]],
					vdata[tindices[i][3]], depth);
		}
		CameraPos_temp.push_back(CameraPos[0]);
		for (int j = 1; j<int(CameraPos.size()); j++)
			{
				for (int k = 0; k<j; k++)
				{
					if (CameraPos.at(k).x==CameraPos.at(j).x && CameraPos.at(k).y==CameraPos.at(j).y && CameraPos.at(k).z==CameraPos.at(j).z)
						break;
					if(k == j-1)
						CameraPos_temp.push_back(CameraPos[j]);
				}
			}
		CameraPos = CameraPos_temp;
		cout << "View points in total: " << CameraPos.size() << endl;
		cout << "The coordinate of view point: " << endl;
		for(int i=0; i < (int)CameraPos.size(); i++) {
			cout << CameraPos.at(i).x <<' '<< CameraPos.at(i).y << ' ' << CameraPos.at(i).z << endl;
		}
	}
	void IcoSphere::norm(float v[])
	{

		float len = 0;

		for (int i = 0; i < 3; ++i) {
			len += v[i] * v[i];
		}

		len = sqrt(len);

		for (int i = 0; i < 3; ++i) {
			v[i] /= ((float)len);
		}
	}

	void IcoSphere::add(float v[])
	{
		Point3f temp_Campos;
		std::vector<float>* temp = new std::vector<float>;
		for (int k = 0; k < 3; ++k) {
			vertexList.push_back(v[k]);
			vertexNormalsList.push_back(v[k]);
			temp->push_back(v[k]);
		}
		temp_Campos.x = temp->at(0);temp_Campos.y = temp->at(1);temp_Campos.z = temp->at(2);
		CameraPos.push_back(temp_Campos);
	}

	void IcoSphere::subdivide(float v1[], float v2[], float v3[], int depth)
	{

		norm(v1);
		norm(v2);
		norm(v3);
		if (depth == 0) {
			add(v1);
			add(v2);
			add(v3);
			return;
		}

		float* v12 = new float[3];
		float* v23 = new float[3];
		float* v31 = new float[3];

		for (int i = 0; i < 3; ++i) {
			v12[i] = (v1[i] + v2[i]) / 2;
			v23[i] = (v2[i] + v3[i]) / 2;
			v31[i] = (v3[i] + v1[i]) / 2;
		}

		norm(v12);
		norm(v23);
		norm(v31);

		subdivide(v1, v12, v31, depth - 1);
		subdivide(v2, v23, v12, depth - 1);
		subdivide(v3, v31, v23, depth - 1);
		subdivide(v12, v23, v31, depth - 1);
	}


}}
