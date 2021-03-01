#include "opencv2/ptcloud/sac_utils.hpp"


namespace cv {
namespace ptcloud {


// neither of a,b,c may be 0 !
void generatePlane(Mat &cloud, const std::vector<double> &coeffs, int N) {
    int n=sqrt(N);
    for (int i=0; i<n; i++) {
	    for (int j=0; j<n; j++) {
			float z = - (coeffs[0] * j + coeffs[1] * i + coeffs[3]) / coeffs[2];
			cloud.push_back(Point3f(j, i, z));
	    }
	}
}

void generateSphere(Mat &cloud, const std::vector<double> &coeffs, int N) {
    for (int i=0; i<N; i++) {
        float x = theRNG().uniform(-10,10);
        float y = theRNG().uniform(-10,10);
        float z = theRNG().uniform(-10,10);
        double l = sqrt(x*x+y*y+z*z);
        Point3f p(x/l,y/l,z/l);
        p *= coeffs[3];
        p += Point3f(coeffs[0], coeffs[1], coeffs[2]);
        cloud.push_back(p);
    }
}

void generateCylinder(Mat &cloud, const std::vector<double> &coeffs, int N) {
	float R=20;
	Point3f cen(coeffs[0],coeffs[1],coeffs[2]);
	Point3f dir(coeffs[3],coeffs[4],coeffs[5]);
	for (int i=0; i<N; i++) {
		float x = theRNG().uniform(-R,R);
		float y = theRNG().uniform(-R,R);
		float z = theRNG().uniform(-R,R);
		float d = theRNG().uniform(-R,R);
		Point3f n = dir.cross(Point3f(x,y,z));
		n *= coeffs[6] / norm(n);
		n += d * dir;
        cloud.push_back(n + cen);
    }
}

void generateRandom(Mat &cloud, const std::vector<double> &coeffs, int N) {
	float R = coeffs[3];
	for (int i=0; i<N; i++) {
		float x = theRNG().uniform(-R,R);
		float y = theRNG().uniform(-R,R);
		float z = theRNG().uniform(-R,R);
		Point3f p(coeffs[0],coeffs[1],coeffs[2]);
        cloud.push_back(p + Point3f(x,y,z));
    }
}

} // ptcloud
} // cv