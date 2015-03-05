import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.*;
import org.opencv.objdetect.*;
import org.opencv.face.*;
import org.opencv.utils.*;
import java.util.*;
import lowgui.*;
import java.io.File;
import java.io.IOException;
import lowgui.*;


class FaceRec {
    FaceRecognizer fr = Face.createLBPHFaceRecognizer();

    //
    // unlike the c++ demo, let's not mess with csv files, but use a folder on disk.
    //    each person should have its own subdir with images (all images the same size, ofc.)
    //
    public Size loadTrainDir(String dir)
    {
        Size s = null;
        int label = 0;
        List<Mat> images = new ArrayList<Mat>();
        List<java.lang.Integer> labels = new ArrayList<java.lang.Integer>();
        File node = new File(dir);
        String[] subNode = node.list();
        if ( subNode==null ) return null;
        for(String person : subNode) {
            File subDir = new File(node, person);
            if ( ! subDir.isDirectory() ) continue;
            File[] pics = subDir.listFiles();
            for(File f : pics) {
                Mat m = Imgcodecs.imread(f.getAbsolutePath(),0);
                if (! m.empty()) {
                    images.add(m);
                    labels.add(label);
                    fr.setLabelInfo(label,subDir.getName());
                    s = m.size();
                }
            }
            label ++;
        }
        fr.train(images, Converters.vector_int_to_Mat(labels));
        return s;
    }

    public String predict(Mat img)
    {
        int[] id = {-1};
        double[] dist = {-1};
        fr.predict(img,id,dist);
        if (id[0] == -1)
            return "";
        double d = ((int)(dist[0]*100));
        return fr.getLabelInfo(id[0]) + " : " + d/100;
    }
}

//
// SimpleSample [persons_dir] [path/to/face_cascade]
//
class SimpleSample {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String personsDir = "e:/code/opencv_p/face3/persons";
        if (args.length > 1) personsDir = args[1];

        String cascadeFile = "e:/code/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
        if (args.length > 2) cascadeFile = args[2];

        CascadeClassifier cascade = new CascadeClassifier(cascadeFile);
        System.out.println("cascade loaded: "+(!cascade.empty())+" !");

        FaceRec face = new FaceRec();
        Size trainSize = face.loadTrainDir(personsDir);
        System.out.println("facerec trained: "+(trainSize!=null)+" !");

        NamedWindow    frame = new NamedWindow("Face");

        VideoCapture cap = new VideoCapture(0);
        if (! cap.isOpened()) {
            System.out.println("Sorry, we could not open you capture !");
        }

        Mat im = new Mat();
        while (cap.read(im)) {
            Mat gray = new Mat();
            Imgproc.cvtColor(im, gray, Imgproc.COLOR_BGR2GRAY);
            if (cascade != null ) {
                MatOfRect faces = new MatOfRect();
                cascade.detectMultiScale(gray, faces);
                Rect[] facesArray = faces.toArray();
                if (facesArray.length != 0) {
                    Rect found = facesArray[0];
                    Imgproc.rectangle(im, found.tl(), found.br(), new Scalar(0,200,0), 3);

                    Mat fi = gray.submat(found);
                    if (fi.size() != trainSize) // not needed for lbph, but for eigen and fisher
                        Imgproc.resize(fi,fi,trainSize);

                    String s = face.predict(fi);
                    if (s != "")
                        Imgproc.putText(im, s, new Point(40,40), Core.FONT_HERSHEY_PLAIN,1.3,new Scalar(0,0,200),2);
                }
            }
            frame.imshow(im);
            int k = frame.waitKey(30);
            if (k == 27) // 'esc'
                break;
            if (k == 's')
                Imgcodecs.imwrite("frame.png", im);
        }
        System.exit(0); // to break out of the ant shell.
    }
}
