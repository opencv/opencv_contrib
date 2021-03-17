#include "opencv2/ptcloud/sac_utils.hpp"


namespace cv {
namespace ptcloud {


// neither of a,b,c may be 0 !
void generatePlane(Mat &cloud, const std::vector<double> &coeffs, int N) {
    int n=(int)sqrt(N);
    for (int i=0; i<n; i++) {
	    for (int j=0; j<n; j++) {
			double z = - (coeffs[0] * j + coeffs[1] * i + coeffs[3]) / coeffs[2];
			cloud.push_back(Point3f(float(j), float(i), float(z)));
	    }
	}
}

void generateSphere(Mat &cloud, const std::vector<double> &coeffs, int N) {
    for (int i=0; i<N; i++) {
        double R = 10.0;
        double x = theRNG().uniform(-R,R);
        double y = theRNG().uniform(-R,R);
        double z = theRNG().uniform(-R,R);
        double l = sqrt(x*x+y*y+z*z);
        Point3d p(x/l,y/l,z/l);
        p *= coeffs[3];
        p += Point3d(coeffs[0], coeffs[1], coeffs[2]);
        cloud.push_back(Point3f(p));
    }
}

void generateCylinder(Mat &cloud, const std::vector<double> &coeffs, int N) {
	double R=20;
	Point3d cen(coeffs[0],coeffs[1],coeffs[2]);
	Point3d dir(coeffs[3],coeffs[4],coeffs[5]);
	for (int i=0; i<N; i++) {
		double x = theRNG().uniform(-R,R);
		double y = theRNG().uniform(-R,R);
		double z = theRNG().uniform(-R,R);
		double d = theRNG().uniform(-R,R);
		Point3d n = dir.cross(Point3d(x,y,z));
		n *= coeffs[6] / norm(n);
		n += d * dir;
        cloud.push_back(Point3f(n + cen));
    }
}

void generateRandom(Mat &cloud, const std::vector<double> &coeffs, int N) {
	float R = float(coeffs[3]);
	for (int i=0; i<N; i++) {
		float x = theRNG().uniform(-R,R);
		float y = theRNG().uniform(-R,R);
		float z = theRNG().uniform(-R,R);
		Point3d p(coeffs[0],coeffs[1],coeffs[2]);
        cloud.push_back(Point3f(p) + Point3f(x,y,z));
    }
}

} // ptcloud
} // cv