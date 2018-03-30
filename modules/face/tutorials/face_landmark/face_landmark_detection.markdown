Face landmark detection in an image {#tutorial_face_landmark_detection_in_an_image}
===================================

![](images/facereg.jpg)

This application lets you detect landmarks of detected faces in an image. You can detect landmarks of all the faces found in an image
and use them further in various applications like face swapping, face averaging etc.
This functionality is now available in OpenCV.

```
// Command to be typed for running the sample
./sampleDetectLandmarks -file=trained_model.dat -face_cascade=lbpcascadefrontalface.xml -image=/path_to_image/image.jpg
```
### Description of command parameters {tutorial_face_training_parameters}

> * **model_filename** f : (REQUIRED) A path to binary file storing the trained model which is to be loaded [example - /data/file.dat]
> * **image** i : (REQUIRED) A path to image in which face landmarks have to be detected.[example - /data/image.jpg]
> * **face_cascade** c : (REQUIRED) A path to the face cascade xml file which you want to use as a face detector.

Understanding code
------------------

![](images/d.png)

This tutorial will explain the sample code for face landmark detection. Jumping directly to the code :

``` c++
CascadeClassifier face_cascade;
face_cascade.load(cascade_name);

Mat img = imread(image);
Ptr<Facemark> facemark = createFacemarkKazemi());
facemark->loadModel(filename);
cout<<"Loaded model"<<endl;
```

The above code creates a CascadeClassifier to detect face regions, and an instance of the face landmark detection class.
We need to load a pretrained model for face landmark detection, and a cascade file for the face detection.
It also loads the image in which landmarks have to be detected.


``` c++
vector<Rect> faces;
resize(img,img,Size(460,460),0,0,INTER_LINEAR_EXACT);

Mat gray;
std::vector<Rect> faces;
if(img.channels()>1){
    cvtColor(img.getMat(),gray,CV_BGR2GRAY);
}
else{
    gray = img.getMat().clone();
}
equalizeHist( gray, gray );

face_cascade.detectMultiScale( gray, faces, 1.1, 3,0, Size(30, 30) );
```

After doing some preprocessing, we first have to detect possible face regions (which will be stored in a `vector<Rect>`.
Also, the image is resized to a smaller size as processing speed is faster with small images.


``` c++
vector< vector<Point2f> > shapes;

if (facemark->fit(img,faces,shapes))
{
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
    }
    for (unsigned long i=0;i<faces.size();i++){
        for(unsigned long k=0;k<shapes[i].size();k++)
            cv::circle(img,shapes[i][k],5,cv::Scalar(0,0,255),FILLED);
    }
    namedWindow("Detected_shape");
    imshow("Detected_shape",img);
    waitKey(0);
}
```


It then creates a vector of vector to store shapes for each face detected.
The above code calls the function fit to get shapes of all detected faces in the image
and then draws the rectangles bounding the faces and marks the desired landmarks.

### Detection Results

![](ab.jpg)

![](ab-1.jpg)
