#include "../include/cnn_3dobj.hpp"
using namespace cv;
using namespace std;

namespace cv{ namespace cnn_3dobj{

	IcoSphere::IcoSphere(float radius_in, int depth_in)
	{

		X = 0.525731112119133606f;
		Z = 0.850650808352039932f;
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
		cout << "View points in total: " << CameraPos->size() << endl;
		cout << "The coordinate of view point: " << endl;
		for(int i=0; i < CameraPos->size(); i++)
		   {
				   cout << CameraPos->at(i).x  << endl;
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
			v[i] /= ((float)len/(float)IcoSphere::radius);
		}
	}

	void IcoSphere::add(float v[])
	{
		Point3f temp_Campos;
		std::vector<float>* temp = new std::vector<float>;
		for (int k = 0; k < 3; ++k) {
			vertexList->push_back(v[k]);
			vertexNormalsList->push_back(v[k]);
			temp->push_back(v[k]);
		}
		temp_Campos.x = temp->at(0);temp_Campos.y = temp->at(1);temp_Campos.z = temp->at(2);
		CameraPos->push_back(temp_Campos);
	}

	void IcoSphere::subdivide(float v1[], float v2[], float v3[], int depth)
	{

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




