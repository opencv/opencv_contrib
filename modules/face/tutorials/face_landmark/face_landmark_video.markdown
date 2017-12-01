Face landmark detection in a video{#tutorial_face_landmark_detection_in_video}
===================================

This application lets you detect landmarks of detected faces in a video.This application first detects faces in a current video frame
and then finds their facial landmarks. You just have to pass the video as input.
```
// Command to be typed for running the sample
./sampleDetectLandmarks -file=trained_model.dat -face_cascade=lbpcascadefrontalface.xml -video=/path_to_video/video.avi
```
Description of command parameters
---------------------------------

> * **model_filename** f : (REQUIRED) A path to binary file storing the trained model which is to be loaded [example - /data/file.dat]
> * **video** v : (REQUIRED) A path to video in which face landmarks have to be detected.[example - /data/video.avi]
> * **face_cascade** c : (REQUIRED) A path to the face cascade xml file which you want to use as a face detector.

### Understanding code

This tutorial will explain the sample code for face landmark detection. Jumping directly to the code :

``` c++
CascadeClassifier face_cascade;
bool myDetector( InputArray image, OutputArray ROIs );

bool myDetector( InputArray image, OutputArray ROIs ){
    Mat gray;
    std::vector<Rect> faces;
    if(image.channels()>1){
        cvtColor(image.getMat(),gray,CV_BGR2GRAY);
    }
    else{
        gray = image.getMat().clone();
    }
    equalizeHist( gray, gray );
    face_cascade.detectMultiScale( gray, faces, 1.1, 3,0, Size(30, 30) );
    Mat(faces).copyTo(ROIs);
    return true;
}
```
The facemark API provides the functionality to the user to use their own face detector to be used in face landmark detection.The above code creartes a sample face detector. The above function would be passed to a function pointer in the facemark API.

``` c++
VideoCapture cap(video);
if(!cap.isOpened()){
	cerr<<"Video cannot be loaded. Give correct path"<<endl;
	return -1;
}
```

The above code creates a video capture object and then loads the video.
If the video is not loaded properly it prompts the user else the code proceeds.

``` c++
Mat img = imread(image);
face_cascade.load(cascade_name);
FacemarkKazemi::Params params;
params.configfile = configfile_name;
Ptr<Facemark> facemark = FacemarkKazemi::create(params);
facemark->setFaceDetector(myDetector);

```
The above code creates a pointer of the face landmark detection class. The face detector created above has to be passed
as function pointer to the facemark pointer created for detecting faces.
``` c++
vector<Rect> faces;
vector< vector<Point2f> > shapes;
Mat img;
```
The above code creates a vector to store the detected faces and a vector of vector to store shapes for each
face detected in the current frame.

``` c++
while(1){
	faces.clear();
	shapes.clear();
	cap>>img;
	resize(img,img,Size(600,600),0,0,INTER_LINEAR_EXACT);
	facemark->getFaces(img,faces);
	if(faces.size()==0){
	    cout<<"No faces found in this frame"<<endl;
	}
	else{
	    for( size_t i = 0; i < faces.size(); i++ )
	    {
		cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
	    }
	    if(facemark->fit(img,faces,shapes))
	    {
		for(unsigned long i=0;i<faces.size();i++){
		    for(unsigned long k=0;k<shapes[i].size();k++)
		        cv::circle(img,shapes[i][k],3,cv::Scalar(0,0,255),FILLED);
		}
	    }
	}
	namedWindow("Detected_shape");
	imshow("Detected_shape",img);
	if(waitKey(1) >= 0) break;
}
```

The above code then reads each frame and detects faces and the landmarks corresponding to each shape detected.
It then displays the current frame.

After running the above code you will get results something like this

Sample video:

@htmlonly
<iframe width="560" height="315" src="https://www.youtube.com/embed/ZtaV07T90D8" frameborder="0" allowfullscreen></iframe>
@endhtmlonly