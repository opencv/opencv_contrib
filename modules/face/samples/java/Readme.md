### a simple usage example for facerecognition in java

<pre>
a build.xml file for ant is supplied, use it like:

    ant -DocvJarDir=/path/to/opencv_300.jar -DocvLibDir=/path/to/opencv/dlls %*
    
on my (win) box, this is:

    ant -DocvJarDir=E:\code\opencv\build\bin -DocvLibDir=E:\code\opencv\build\bin\Release %*
    

additional args are: 

    * path to persons folder (images, see below)
    * path to face-cascade (e.g. haarcascade_frontalface.xml)

-----------------------------------------------------------------------

unlike the c++ demo, this one does not use csv files for training, but a folder on disk.
each person should have its own subdir with images (all images the same size, ofc.)

    /persons
       /peter
         img0.png
         img17.jpg
         img23.png
       /paul
         img0.png
         img1.jpg
         img2.png
        /mary
         img0.png
         img1.jpg
         img2.png
    
</pre>
