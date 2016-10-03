
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.core;

import java.lang.String;

// C++: class Algorithm
//javadoc: Algorithm
public class Algorithm {

    protected final long nativeObj;
    protected Algorithm(long addr) { nativeObj = addr; }


    //
    // C++:  String getDefaultName()
    //

    //javadoc: Algorithm::getDefaultName()
    public  String getDefaultName()
    {
        
        String retVal = getDefaultName_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void clear()
    //

    //javadoc: Algorithm::clear()
    public  void clear()
    {
        
        clear_0(nativeObj);
        
        return;
    }


    //
    // C++:  void save(String filename)
    //

    //javadoc: Algorithm::save(filename)
    public  void save(String filename)
    {
        
        save_0(nativeObj, filename);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  String getDefaultName()
    private static native String getDefaultName_0(long nativeObj);

    // C++:  void clear()
    private static native void clear_0(long nativeObj);

    // C++:  void save(String filename)
    private static native void save_0(long nativeObj, String filename);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
