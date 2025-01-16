![](images/2.jpg)

Training face landmark detector{#tutorial_face_training_face_landmark_detector}
==============================

This application helps to train your own face landmark detector. You can train your own face landmark detection by just providing the paths for
directory containing the images and files containing their corresponding face landmarks. As this landmark detector was originally trained on
[HELEN dataset](http://www.ifp.illinois.edu/~vuongle2/helen/), the training follows the format of data provided in HELEN dataset.

The dataset consists of .txt files whose first line contains the image name which then follows the annotations.
The format of the file containing annotations should be of following format :
>       /directory/images/abc.jpg
>       123.45,345.65
>       321.67,543.89
>       .... , ....
>       .... , ....
The above format is similar to HELEN dataset which is used for training the model.

```
// Command to be typed for running the sample
./sample_train_landmark_detector -annotations=/home/sukhad/Downloads/code/trainset/ -config=config.xml -face_cascade=lbpcascadefrontalface.xml -model=trained_model.dat -width=460 -height=460
```

## Description of command parameters

> * **annotations** a : (REQUIRED) Path to annotations txt file [example - /data/annotations.txt]
> * **config** c : (REQUIRED) Path to configuration xml file containing parameters for training.[ example - /data/config.xml]
> * **model** m :  (REQUIRED) Path to configuration xml file containing parameters for training.[ example - /data/model.dat]
> * **width** w : (OPTIONAL)  The width which you want all images to get to scale the annotations. Large images are slow to process [default = 460]
> * **height** h : (OPTIONAL) The height which you want all images to get to scale the annotations. Large images are slow to process [default = 460]
> * **face_cascade** f (REQUIRED) Path to the face cascade xml file which you want to use as a detector.

## Description of training parameters


The configuration file described above which is used while training contains the training parameters which are required for training.

**The description of parameters is as follows :**

1. **Cascade depth :** This stores the depth of cascade of regressors used for training.
2. **Tree depth :** This stores the depth of trees created as weak learners during gradient boosting.
3. **Number of trees per cascade level :** This stores number of trees required per cascade level.
4. **Learning rate :** This stores the learning rate for gradient boosting.This is required to prevent overfitting using shrinkage.
5. **Oversampling amount :** This stores the oversampling amount for the samples.
6. **Number of test coordinates :** This stores number of test coordinates to be generated as samples to decide for making the split.
7. **Lambda :** This stores the value used for calculating the probabilty which helps to select closer pixels for making the split.
8. **Number of test splits :** This stores the number of test splits to be generated before making the best split.


To get more detailed description about the training parameters you can refer to the [Research paper](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf).

## Understanding code


![](images/3.jpg)


Jumping directly to the code :

``` c++
CascadeClassifier face_cascade;
bool myDetector( InputArray image, OutputArray ROIs );

bool myDetector( InputArray image, OutputArray ROIs ){
    Mat gray;
    std::vector<Rect> faces;
    if(image.channels()>1){
        cvtColor(image.getMat(),gray,COLOR_BGR2GRAY);
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
The facemark API provides the functionality to the user to use their own face detector to be used in training.The above code creartes a sample face detector. The above function would be passed to a function pointer in the facemark API.

``` c++
vector<String> filenames;
glob(directory,filenames);
```
The above code creates a vector filenames for storing the names of the .txt files.
It gets the filenames of the files in the directory.

``` c++
Mat img = imread(image);
face_cascade.load(cascade_name);
FacemarkKazemi::Params params;
params.configfile = configfile_name;
Ptr<Facemark> facemark = FacemarkKazemi::create(params);
facemark->setFaceDetector(myDetector);

```
The above code creates a pointer of the face landmark detection class. The face detector created above has to be passed
as function pointer to the facemark pointer created for detecting faces while training the model.

``` c++
vector<String> imagenames;
vector< vector<Point2f> > trainlandmarks,Trainlandmarks;
vector<Mat> trainimages;
loadTrainingData(filenames,trainlandmarks,imagenames);
for(unsigned long i=0;i<300;i++){
string imgname = imagenames[i].substr(0, imagenames[i].size()-1);
string img = directory + string(imgname) + ".jpg";
Mat src = imread(img);
if(src.empty()){
    cerr<<string("Image "+img+" not found\n.")<<endl;
    continue;
}
trainimages.push_back(src);
Trainlandmarks.push_back(trainlandmarks[i]);
}
```
The above code creates std::vectors to store the images and their corresponding landmarks.
The above code calls a function loadTrainingData to load the landmarks and the images into their respective vectors.

If the dataset you downloaded is of the following format :
```
version: 1
n_points:  68
{
 115.167660 220.807529
 116.164839 245.721357
 120.208690 270.389841
  ...
}
This is the example of the dataset available at https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

```

Then skip the above code for loading training data and use the following code. This sample is provided as sampleTrainLandmarkDetector2.cpp
in the face module in opencv contrib.

``` c++
std::vector<String> images;
std::vector<std::vector<Point2f> > facePoints;
loadTrainingData(imagesList, annotations, images, facePoints, 0.0);
```

In the above code imagelist and annotations are the file of following format :
```
example of contents for images.txt:
../trainset/image_0001.png
../trainset/image_0002.png
example of contents for annotation.txt:
../trainset/image_0001.pts
../trainset/image_0002.pts
```

These symbolize the names of images and their corresponding annotations.

The above code scales images and landmarks as training on images of smaller size takes less time.
This is because processing larger images requires more time. After scaling data it calculates mean
shape of the data which is used as initial shape while training.

Finally call the following function to perform training :

``` c++
facemark->training(Trainimages,Trainlandmarks,configfile_name,scale,modelfile_name);
```
In the above function scale is passed to scale all images and the corresponding landmarks so that the size of all
images can be reduced as it takes greater time to process large images.
This call to the train function trains the model and stores the trained model file with the given
filename specified.As the training starts successfully you will see something like this :
![](images/train1.png)


**The error rate on trained images depends on the number of images used for training used as follows :**

![](images/train.png)

**The error rate on test images depends on the number of images used for training used as follows :**

![](images/test.png)
