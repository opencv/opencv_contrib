
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.videoio;

import java.lang.String;
import org.opencv.core.Mat;

// C++: class VideoCapture
//javadoc: VideoCapture
public class VideoCapture {

    protected final long nativeObj;
    protected VideoCapture(long addr) { nativeObj = addr; }


    //
    // C++:   VideoCapture(String filename, int apiPreference)
    //

    //javadoc: VideoCapture::VideoCapture(filename, apiPreference)
    public   VideoCapture(String filename, int apiPreference)
    {
        
        nativeObj = VideoCapture_0(filename, apiPreference);
        
        return;
    }


    //
    // C++:   VideoCapture(String filename)
    //

    //javadoc: VideoCapture::VideoCapture(filename)
    public   VideoCapture(String filename)
    {
        
        nativeObj = VideoCapture_1(filename);
        
        return;
    }


    //
    // C++:   VideoCapture(int index)
    //

    //javadoc: VideoCapture::VideoCapture(index)
    public   VideoCapture(int index)
    {
        
        nativeObj = VideoCapture_2(index);
        
        return;
    }


    //
    // C++:   VideoCapture()
    //

    //javadoc: VideoCapture::VideoCapture()
    public   VideoCapture()
    {
        
        nativeObj = VideoCapture_3();
        
        return;
    }


    //
    // C++:  bool grab()
    //

    //javadoc: VideoCapture::grab()
    public  boolean grab()
    {
        
        boolean retVal = grab_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool isOpened()
    //

    //javadoc: VideoCapture::isOpened()
    public  boolean isOpened()
    {
        
        boolean retVal = isOpened_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool open(String filename, int apiPreference)
    //

    //javadoc: VideoCapture::open(filename, apiPreference)
    public  boolean open(String filename, int apiPreference)
    {
        
        boolean retVal = open_0(nativeObj, filename, apiPreference);
        
        return retVal;
    }


    //
    // C++:  bool open(String filename)
    //

    //javadoc: VideoCapture::open(filename)
    public  boolean open(String filename)
    {
        
        boolean retVal = open_1(nativeObj, filename);
        
        return retVal;
    }


    //
    // C++:  bool open(int index)
    //

    //javadoc: VideoCapture::open(index)
    public  boolean open(int index)
    {
        
        boolean retVal = open_2(nativeObj, index);
        
        return retVal;
    }


    //
    // C++:  bool read(Mat& image)
    //

    //javadoc: VideoCapture::read(image)
    public  boolean read(Mat image)
    {
        
        boolean retVal = read_0(nativeObj, image.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool retrieve(Mat& image, int flag = 0)
    //

    //javadoc: VideoCapture::retrieve(image, flag)
    public  boolean retrieve(Mat image, int flag)
    {
        
        boolean retVal = retrieve_0(nativeObj, image.nativeObj, flag);
        
        return retVal;
    }

    //javadoc: VideoCapture::retrieve(image)
    public  boolean retrieve(Mat image)
    {
        
        boolean retVal = retrieve_1(nativeObj, image.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool set(int propId, double value)
    //

    //javadoc: VideoCapture::set(propId, value)
    public  boolean set(int propId, double value)
    {
        
        boolean retVal = set_0(nativeObj, propId, value);
        
        return retVal;
    }


    //
    // C++:  double get(int propId)
    //

    //javadoc: VideoCapture::get(propId)
    public  double get(int propId)
    {
        
        double retVal = get_0(nativeObj, propId);
        
        return retVal;
    }


    //
    // C++:  void release()
    //

    //javadoc: VideoCapture::release()
    public  void release()
    {
        
        release_0(nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   VideoCapture(String filename, int apiPreference)
    private static native long VideoCapture_0(String filename, int apiPreference);

    // C++:   VideoCapture(String filename)
    private static native long VideoCapture_1(String filename);

    // C++:   VideoCapture(int index)
    private static native long VideoCapture_2(int index);

    // C++:   VideoCapture()
    private static native long VideoCapture_3();

    // C++:  bool grab()
    private static native boolean grab_0(long nativeObj);

    // C++:  bool isOpened()
    private static native boolean isOpened_0(long nativeObj);

    // C++:  bool open(String filename, int apiPreference)
    private static native boolean open_0(long nativeObj, String filename, int apiPreference);

    // C++:  bool open(String filename)
    private static native boolean open_1(long nativeObj, String filename);

    // C++:  bool open(int index)
    private static native boolean open_2(long nativeObj, int index);

    // C++:  bool read(Mat& image)
    private static native boolean read_0(long nativeObj, long image_nativeObj);

    // C++:  bool retrieve(Mat& image, int flag = 0)
    private static native boolean retrieve_0(long nativeObj, long image_nativeObj, int flag);
    private static native boolean retrieve_1(long nativeObj, long image_nativeObj);

    // C++:  bool set(int propId, double value)
    private static native boolean set_0(long nativeObj, int propId, double value);

    // C++:  double get(int propId)
    private static native double get_0(long nativeObj, int propId);

    // C++:  void release()
    private static native void release_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
