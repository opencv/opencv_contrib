Using the FacemarkAAM {#tutorial_facemark_aam}
==========================================================

Goals
----

In this tutorial you will learn how to:
- creating the instance of FacemarkAAM
- training the AAM model
- Fitting using FacemarkAAM

Preparation
--------

Before you continue with this tutorial, you should download the dataset of facial landmarks detection.
We suggest you to download the LFPW dataset which can be retrieved at <https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip>.

Make sure that the annotation format is supported by the API, the contents in annotation file should look like the following snippet:
@code
version: 1
n_points:  68
{
212.716603 499.771793
230.232816 566.290071
...
}
@endcode

The next thing to do is to make 2 text files containing the list of image files and annotation files respectively. Make sure that the order or image and annotation in both files are matched. Furthermore, it is advised to use absolute path instead of relative path.
Example to make the file list in Linux machine
@code
ls $PWD/trainset/*.jpg > images_train.txt
ls $PWD/trainset/*.pts > annotation_train.txt
@endcode

example of content in the images_train.txt
@code
/home/user/lfpw/trainset/100032540_1.jpg
/home/user/lfpw/trainset/100040721_1.jpg
/home/user/lfpw/trainset/100040721_2.jpg
/home/user/lfpw/trainset/1002681492_1.jpg
@endcode

example of content in the annotation_train.txt
@code
/home/user/lfpw/trainset/100032540_1.pts
/home/user/lfpw/trainset/100040721_1.pts
/home/user/lfpw/trainset/100040721_2.pts
/home/user/lfpw/trainset/1002681492_1.pts
@endcode

Optionally, you can create the similar files for the testset.

In this tutorial, the pre-trained model will not be provided due to its large file size (~500MB). By following this tutorial, you will be able to train obtain your own trained model within few minutes.

Working with the AAM Algorithm
--------

The full working code is available in the face/samples/facemark_demo_aam.cpp file. In this tutorial, the explanation of some important parts are covered.

-# <B>Creating the instance of AAM algorithm</B>

   @snippet face/samples/facemark_demo_aam.cpp instance_creation
   Firstly, an instance of parameter for the AAM algorithm is created. In this case, we will modify the default list of the scaling factor. By default, the scaling factor used is 1.0 (no scaling). Here we add two more scaling factor which will make the instance trains two more model at scale 2 and 4 (2 time smaller and 4 time smaller, faster faster fitting time). However, you should make sure that this scaling factor is not too big since it will make the image scaled into a very small one. Thus it will lost all of its important information for the landmark detection purpose.

   Alternatively, you can override the default scaling in similar way to this example:
   @code
   std::vector<float>scales;
   scales.push_back(1.5);
   scales.push_back(2.4);

   FacemarkAAM::Params params;
   params.scales = scales;
   @endcode

-# <B>Loading the dataset</B>

   @snippet face/samples/facemark_demo_aam.cpp load_dataset
   List of the dataset are loaded into the program. We will put the samples from dataset one by one in the next step.

-# <B>Adding the samples to the trainer</B>

   @snippet face/samples/facemark_demo_aam.cpp add_samples
   The image from the dataset list are loaded one by one as well as its corresponding annotation data. Then the pair of sample is added to the trainer.

-# <B>Training process</B>

   @snippet face/samples/facemark_demo_aam.cpp training
   The training process is called using a single line of code. Make sure that all the required training samples are already added to the trainer.

-# <B>Preparation for fitting</B>

   First of all, you need to load the list of test files.
   @snippet face/samples/facemark_demo_aam.cpp load_test_images

   Since the AAM needs initialization parameters (rotation, translation, and scaling), you need to declare the required variable to store these information which will be obtained using a custom function. Since the implementation of getInitialFitting() function in this example is not optimal, you can create your own function.

   The initialization is obtained by comparing the base shape of the trained model with the current face image. In this case, the rotation is obtained by comparing the angle of line formed by two eyes in the input face image with the same line in the base shape. Meanwhile, the scaling is obtained by comparing the length of line between eyes in the input image compared to the base shape.

-# <B>Fitting process</B>

   The fitting process is started by detecting the face in a given image.
   @snippet face/samples/facemark_demo_aam.cpp detect_face

   If at least one face is found, then the next step is computing the initialization parameters. In this case, since the getInitialFitting() function is not optimal, it may not find pair of eyes from a given face. Therefore, we will filter out the face without initialization parameters and in this case, each element in the `conf` vector represent the initialization parameter for each filtered face.
   @snippet face/samples/facemark_demo_aam.cpp get_initialization

   For the fitting parameter stored in the `conf` vector, the last parameter represent the ID of scaling factor that will be used in the fitting process. In this example the fitting will use the biggest scaling factor (4) which is expected to have the fastest computation time compared to the other scales. If the ID if bigger than the available trained scale in the model, the the model with the biggest scale ID is used.

   The fitting process is quite simple, you just need to put the corresponding image, vector of `cv::Rect` representing the ROIs of all faces in the given image, container of the landmark points represented by `landmarks` variable, and the configuration variables.
   @snippet face/samples/facemark_demo_aam.cpp fitting_process

   After the fitting process is finished, you can visualize the result using the `drawFacemarks` function.
