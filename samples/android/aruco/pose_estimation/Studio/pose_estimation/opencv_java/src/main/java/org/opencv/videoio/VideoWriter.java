
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.videoio;

import java.lang.String;
import org.opencv.core.Mat;
import org.opencv.core.Size;

// C++: class VideoWriter
//javadoc: VideoWriter
public class VideoWriter {

    protected final long nativeObj;
    protected VideoWriter(long addr) { nativeObj = addr; }


    //
    // C++:   VideoWriter(String filename, int fourcc, double fps, Size frameSize, bool isColor = true)
    //

    //javadoc: VideoWriter::VideoWriter(filename, fourcc, fps, frameSize, isColor)
    public   VideoWriter(String filename, int fourcc, double fps, Size frameSize, boolean isColor)
    {
        
        nativeObj = VideoWriter_0(filename, fourcc, fps, frameSize.width, frameSize.height, isColor);
        
        return;
    }

    //javadoc: VideoWriter::VideoWriter(filename, fourcc, fps, frameSize)
    public   VideoWriter(String filename, int fourcc, double fps, Size frameSize)
    {
        
        nativeObj = VideoWriter_1(filename, fourcc, fps, frameSize.width, frameSize.height);
        
        return;
    }


    //
    // C++:   VideoWriter()
    //

    //javadoc: VideoWriter::VideoWriter()
    public   VideoWriter()
    {
        
        nativeObj = VideoWriter_2();
        
        return;
    }


    //
    // C++:  bool isOpened()
    //

    //javadoc: VideoWriter::isOpened()
    public  boolean isOpened()
    {
        
        boolean retVal = isOpened_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool open(String filename, int fourcc, double fps, Size frameSize, bool isColor = true)
    //

    //javadoc: VideoWriter::open(filename, fourcc, fps, frameSize, isColor)
    public  boolean open(String filename, int fourcc, double fps, Size frameSize, boolean isColor)
    {
        
        boolean retVal = open_0(nativeObj, filename, fourcc, fps, frameSize.width, frameSize.height, isColor);
        
        return retVal;
    }

    //javadoc: VideoWriter::open(filename, fourcc, fps, frameSize)
    public  boolean open(String filename, int fourcc, double fps, Size frameSize)
    {
        
        boolean retVal = open_1(nativeObj, filename, fourcc, fps, frameSize.width, frameSize.height);
        
        return retVal;
    }


    //
    // C++:  bool set(int propId, double value)
    //

    //javadoc: VideoWriter::set(propId, value)
    public  boolean set(int propId, double value)
    {
        
        boolean retVal = set_0(nativeObj, propId, value);
        
        return retVal;
    }


    //
    // C++:  double get(int propId)
    //

    //javadoc: VideoWriter::get(propId)
    public  double get(int propId)
    {
        
        double retVal = get_0(nativeObj, propId);
        
        return retVal;
    }


    //
    // C++: static int fourcc(char c1, char c2, char c3, char c4)
    //

    //javadoc: VideoWriter::fourcc(c1, c2, c3, c4)
    public static int fourcc(char c1, char c2, char c3, char c4)
    {
        
        int retVal = fourcc_0(c1, c2, c3, c4);
        
        return retVal;
    }


    //
    // C++:  void release()
    //

    //javadoc: VideoWriter::release()
    public  void release()
    {
        
        release_0(nativeObj);
        
        return;
    }


    //
    // C++:  void write(Mat image)
    //

    //javadoc: VideoWriter::write(image)
    public  void write(Mat image)
    {
        
        write_0(nativeObj, image.nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   VideoWriter(String filename, int fourcc, double fps, Size frameSize, bool isColor = true)
    private static native long VideoWriter_0(String filename, int fourcc, double fps, double frameSize_width, double frameSize_height, boolean isColor);
    private static native long VideoWriter_1(String filename, int fourcc, double fps, double frameSize_width, double frameSize_height);

    // C++:   VideoWriter()
    private static native long VideoWriter_2();

    // C++:  bool isOpened()
    private static native boolean isOpened_0(long nativeObj);

    // C++:  bool open(String filename, int fourcc, double fps, Size frameSize, bool isColor = true)
    private static native boolean open_0(long nativeObj, String filename, int fourcc, double fps, double frameSize_width, double frameSize_height, boolean isColor);
    private static native boolean open_1(long nativeObj, String filename, int fourcc, double fps, double frameSize_width, double frameSize_height);

    // C++:  bool set(int propId, double value)
    private static native boolean set_0(long nativeObj, int propId, double value);

    // C++:  double get(int propId)
    private static native double get_0(long nativeObj, int propId);

    // C++: static int fourcc(char c1, char c2, char c3, char c4)
    private static native int fourcc_0(char c1, char c2, char c3, char c4);

    // C++:  void release()
    private static native void release_0(long nativeObj);

    // C++:  void write(Mat image)
    private static native void write_0(long nativeObj, long image_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
