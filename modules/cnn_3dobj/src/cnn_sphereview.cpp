#include "precomp.hpp"
using namespace cv;
using namespace std;

namespace cv
{
namespace cnn_3dobj
{
	IcoSphere::IcoSphere(float radius_in, int depth_in)
	{

		X = 0.5f;
		Z = 0.5f;
		X *= (int)radius_in;
		Z *= (int)radius_in;
		diff = 0.00000005964;
		float vdata[12][3] = { { -X, 0.0f, Z }, { X, 0.0f, Z },
				{ -X, 0.0f, -Z }, { X, 0.0f, -Z }, { 0.0f, Z, X }, { 0.0f, Z, -X },
				{ 0.0f, -Z, X }, { 0.0f, -Z, -X }, { Z, X, 0.0f }, { -Z, X, 0.0f },
				{ Z, -X, 0.0f }, { -Z, -X, 0.0f } };


		int tindices[20][3] = { { 0, 4, 1 }, { 0, 9, 4 }, { 9, 5, 4 },
				{ 4, 5, 8 }, { 4, 8, 1 }, { 8, 10, 1 }, { 8, 3, 10 }, { 5, 3, 8 },
				{ 5, 2, 3 }, { 2, 7, 3 }, { 7, 10, 3 }, { 7, 6, 10 }, { 7, 11, 6 },
				{ 11, 0, 6 }, { 0, 1, 6 }, { 6, 1, 10 }, { 9, 0, 11 },
				{ 9, 11, 2 }, { 9, 2, 5 }, { 7, 2, 11 } };

		// Iterate over points
		for (int i = 0; i < 20; ++i) {

			subdivide(vdata[tindices[i][0]], vdata[tindices[i][1]],
					vdata[tindices[i][2]], depth_in);
		}
		CameraPos_temp.push_back(CameraPos[0]);
		for (int j = 1; j<int(CameraPos.size()); j++)
			{
				for (int k = 0; k<j; k++)
				{
					if (CameraPos.at(k).x-CameraPos.at(j).x<diff && CameraPos.at(k).y-CameraPos.at(j).y<diff && CameraPos.at(k).z-CameraPos.at(j).z<diff)
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
	};
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
	};

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
	};

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
	};

	uint32_t IcoSphere::swap_endian(uint32_t val)
	{
		val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
		return (val << 16) | (val >> 16);
	};

	cv::Point3d IcoSphere::getCenter(cv::Mat cloud)
	{
		Point3f* data = cloud.ptr<cv::Point3f>();
		Point3d dataout;
		for(int i = 0; i < cloud.cols; ++i)
		    {
			dataout.x += data[i].x;
			dataout.y += data[i].y;
			dataout.z += data[i].z;
		    }
		dataout.x = dataout.x/cloud.cols;
		dataout.y = dataout.y/cloud.cols;
		dataout.z = dataout.z/cloud.cols;
		return dataout;
	};

	float IcoSphere::getRadius(cv::Mat cloud, cv::Point3d center)
	{
		float radiusCam = 0;
		Point3f* data = cloud.ptr<cv::Point3f>();
		Point3d datatemp;
		for(int i = 0; i < cloud.cols; ++i)
		    {
			datatemp.x = data[i].x - (float)center.x;
			datatemp.y = data[i].y - (float)center.y;
			datatemp.z = data[i].z - (float)center.z;
			float Radius = sqrt(pow(datatemp.x,2)+pow(datatemp.y,2)+pow(datatemp.z,2));
			if(Radius > radiusCam)
			{
				radiusCam = Radius;
			}
		    }
		radiusCam *= 4;
		return radiusCam;
	};

	void IcoSphere::createHeader(int num_item, int rows, int cols, const char* headerPath)
	{
		char* a0 = (char*)malloc(1024);
		strcpy(a0, headerPath);
		char a1[] = "image";
		char a2[] = "label";
		char* headerPathimg = (char*)malloc(1024);
		strcpy(headerPathimg, a0);
		strcat(headerPathimg, a1);
		char* headerPathlab = (char*)malloc(1024);
		strcpy(headerPathlab, a0);
		strcat(headerPathlab, a2);
		std::ofstream headerImg(headerPathimg, ios::out|ios::binary);
		std::ofstream headerLabel(headerPathlab, ios::out|ios::binary);
		int headerimg[4] = {2051,num_item,rows,cols};
		for (int i=0; i<4; i++)
			headerimg[i] = swap_endian(headerimg[i]);
		int headerlabel[2] = {2049,num_item};
		for (int i=0; i<2; i++)
			headerlabel[i] = swap_endian(headerlabel[i]);
		headerImg.write(reinterpret_cast<const char*>(headerimg), sizeof(int)*4);
		headerImg.close();
		headerLabel.write(reinterpret_cast<const char*>(headerlabel), sizeof(int)*2);
		headerLabel.close();
	};

	void IcoSphere::writeBinaryfile(string filenameImg, const char* binaryPath, const char* headerPath, int num_item, int label_class)
	{
		int isrgb = 0;
		cv::Mat ImgforBin = cv::imread(filenameImg, isrgb);
		char* A0 = (char*)malloc(1024);
		strcpy(A0, binaryPath);
		char A1[] = "image";
		char A2[] = "label";
		char* binPathimg = (char*)malloc(1024);
		strcpy(binPathimg, A0);
		strcat(binPathimg, A1);
		char* binPathlab = (char*)malloc(1024);
		strcpy(binPathlab, A0);
		strcat(binPathlab, A2);
		fstream img_file, lab_file;
		img_file.open(binPathimg,ios::in);
		lab_file.open(binPathlab,ios::in);
		if(!img_file)
		{
			cout << "Creating the training data at: " << binaryPath << ". " << endl;
			char* a0 = (char*)malloc(1024);
			strcpy(a0, headerPath);
			char a1[] = "image";
			char a2[] = "label";
			char* headerPathimg = (char*)malloc(1024);
			strcpy(headerPathimg, a0);
			strcat(headerPathimg,a1);
			char* headerPathlab = (char*)malloc(1024);
			strcpy(headerPathlab, a0);
			strcat(headerPathlab,a2);
			createHeader(num_item, 64, 64, binaryPath);
			img_file.open(binPathimg,ios::out|ios::binary|ios::app);
			lab_file.open(binPathlab,ios::out|ios::binary|ios::app);
			for (int r = 0; r < ImgforBin.rows; r++)
			  {
				img_file.write(reinterpret_cast<const char*>(ImgforBin.ptr(r)), ImgforBin.cols*ImgforBin.elemSize());
			  }
			unsigned char templab = (unsigned char)label_class;
			lab_file << templab;
		}
		else
		{
			img_file.close();
			lab_file.close();
			img_file.open(binPathimg,ios::out|ios::binary|ios::app);
			lab_file.open(binPathlab,ios::out|ios::binary|ios::app);
			cout <<"Concatenating the training data at: " << binaryPath << ". " << endl;
			for (int r = 0; r < ImgforBin.rows; r++)
			  {
				img_file.write(reinterpret_cast<const char*>(ImgforBin.ptr(r)), ImgforBin.cols*ImgforBin.elemSize());
			  }
			unsigned char templab = (unsigned char)label_class;
			lab_file << templab;
		}
		img_file.close();
		lab_file.close();
	};
}}
