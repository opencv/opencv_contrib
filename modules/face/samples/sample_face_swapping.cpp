#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp" // seamlessClone()
#include <iostream>
using namespace cv;
using namespace cv::face;
using namespace std;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

void divideIntoTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri);
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &triangle1, vector<Point2f> &triangle2);

//Divide the face into triangles for warping
void divideIntoTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &Tri){

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f triangle = triangleList[i];
        pt[0] = Point2f(triangle[0], triangle[1]);
        pt[1] = Point2f(triangle[2], triangle[3]);
        pt[2] = Point2f(triangle[4], triangle[5]);
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] =(int) k;
            Tri.push_back(ind);
        }
    }
}
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &triangle1, vector<Point2f> &triangle2)
{
    Rect rectangle1 = boundingRect(triangle1);
    Rect rectangle2 = boundingRect(triangle2);
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> triangle1Rect, triangle2Rect;
    vector<Point> triangle2RectInt;
    for(int i = 0; i < 3; i++)
    {
        triangle1Rect.push_back( Point2f( triangle1[i].x - rectangle1.x, triangle1[i].y - rectangle1.y) );
        triangle2Rect.push_back( Point2f( triangle2[i].x - rectangle2.x, triangle2[i].y - rectangle2.y) );
        triangle2RectInt.push_back( Point((int)(triangle2[i].x - rectangle2.x),(int) (triangle2[i].y - rectangle2.y))); // for fillConvexPoly
    }
    // Get mask by filling triangle
    Mat mask = Mat::zeros(rectangle2.height, rectangle2.width, CV_32FC3);
    fillConvexPoly(mask, triangle2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    img1(rectangle1).copyTo(img1Rect);
    Mat img2Rect = Mat::zeros(rectangle2.height, rectangle2.width, img1Rect.type());
    Mat warp_mat = getAffineTransform(triangle1Rect, triangle2Rect);
    warpAffine( img1Rect, img2Rect, warp_mat, img2Rect.size(), INTER_LINEAR, BORDER_REFLECT_101);
    multiply(img2Rect,mask, img2Rect);
    multiply(img2(rectangle2), Scalar(1.0,1.0,1.0) - mask, img2(rectangle2));
    img2(rectangle2) = img2(rectangle2) + img2Rect;
}
int main( int argc, char** argv)
{
    //Give the path to the directory containing all the files containing data
    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | give the following arguments in following format }"
        "{ image1 i1      |      | (required) path to the first image file in which you want to apply swapping }"
        "{ image2 i2      |      | (required) path to the second image file in which you want to apply face swapping }"
        "{ model m        |      | (required) path to the file containing model to be loaded for face landmark detection}"
        "{ face_cascade f |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    Mat img1=imread(parser.get<string>("image1"));
    Mat img2=imread(parser.get<string>("image2"));
    if (img1.empty()||img2.empty()){
        if(img1.empty()){
            parser.printMessage();
            cerr << parser.get<string>("image1")<<" not found" << endl;
            return -1;
        }
        if (img2.empty()){
            parser.printMessage();
            cerr << parser.get<string>("image2")<<" not found" << endl;
            return -1;
        }
    }
    string modelfile_name(parser.get<string>("model"));
    if (modelfile_name.empty()){
        parser.printMessage();
        cerr << "Model file name not found." << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "The name of the cascade classifier to be loaded to detect faces is not found" << endl;
        return -1;
    }
    //create a pointer to call the base class
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);
    facemark->loadModel(modelfile_name);
    cout<<"Loaded model"<<endl;
    //vector to store the faces detected in the image
    vector<Rect> faces1,faces2;
    vector< vector<Point2f> > shape1,shape2;
    //Detect faces in the current image
    float ratio1 = (float)img1.cols/(float)img1.rows;
    float ratio2 = (float)img2.cols/(float)img2.rows;
    resize(img1,img1,Size((int)(640*ratio1),(int)(640*ratio1)), 0, 0, INTER_LINEAR_EXACT);
    resize(img2,img2,Size((int)(640*ratio2),(int)(640*ratio2)), 0, 0, INTER_LINEAR_EXACT);
    Mat img1Warped = img2.clone();
    facemark->getFaces(img1,faces1);
    facemark->getFaces(img2,faces2);
    //Initialise the shape of the faces
    facemark->fit(img1,faces1,shape1);
    facemark->fit(img2,faces2,shape2);
    unsigned long numswaps = (unsigned long)min((unsigned long)shape1.size(),(unsigned long)shape2.size());
    for(unsigned long z=0;z<numswaps;z++){
        vector<Point2f> points1 = shape1[z];
        vector<Point2f> points2 = shape2[z];
        img1.convertTo(img1, CV_32F);
        img1Warped.convertTo(img1Warped, CV_32F);
        // Find convex hull
        vector<Point2f> boundary_image1;
        vector<Point2f> boundary_image2;
        vector<int> index;
        convexHull(Mat(points2),index, false, false);
        for(size_t i = 0; i < index.size(); i++)
        {
            boundary_image1.push_back(points1[index[i]]);
            boundary_image2.push_back(points2[index[i]]);
        }
        // Triangulation for points on the convex hull
        vector< vector<int> > triangles;
        Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
        divideIntoTriangles(rect, boundary_image2, triangles);
        // Apply affine transformation to Delaunay triangles
        for(size_t i = 0; i < triangles.size(); i++)
        {
            vector<Point2f> triangle1, triangle2;
            // Get points for img1, img2 corresponding to the triangles
            for(int j = 0; j < 3; j++)
            {
                triangle1.push_back(boundary_image1[triangles[i][j]]);
                triangle2.push_back(boundary_image2[triangles[i][j]]);
            }
            warpTriangle(img1, img1Warped, triangle1, triangle2);
        }
        // Calculate mask
        vector<Point> hull;
        for(size_t i = 0; i < boundary_image2.size(); i++)
        {
            Point pt((int)boundary_image2[i].x,(int)boundary_image2[i].y);
            hull.push_back(pt);
        }
        Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
        fillConvexPoly(mask,&hull[0],(int)hull.size(), Scalar(255,255,255));
        // Clone seamlessly.
        Rect r = boundingRect(boundary_image2);
        Point center = (r.tl() + r.br()) / 2;
        Mat output;
        img1Warped.convertTo(img1Warped, CV_8UC3);
        seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);
        imshow("Face_Swapped", output);
        waitKey(0);
        destroyAllWindows();
    }
    return 0;
}