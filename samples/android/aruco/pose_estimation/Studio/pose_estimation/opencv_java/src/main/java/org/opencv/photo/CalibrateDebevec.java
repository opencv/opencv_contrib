
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;



// C++: class CalibrateDebevec
//javadoc: CalibrateDebevec
public class CalibrateDebevec extends CalibrateCRF {

    protected CalibrateDebevec(long addr) { super(addr); }


    //
    // C++:  bool getRandom()
    //

    //javadoc: CalibrateDebevec::getRandom()
    public  boolean getRandom()
    {
        
        boolean retVal = getRandom_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getLambda()
    //

    //javadoc: CalibrateDebevec::getLambda()
    public  float getLambda()
    {
        
        float retVal = getLambda_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getSamples()
    //

    //javadoc: CalibrateDebevec::getSamples()
    public  int getSamples()
    {
        
        int retVal = getSamples_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setLambda(float lambda)
    //

    //javadoc: CalibrateDebevec::setLambda(lambda)
    public  void setLambda(float lambda)
    {
        
        setLambda_0(nativeObj, lambda);
        
        return;
    }


    //
    // C++:  void setRandom(bool random)
    //

    //javadoc: CalibrateDebevec::setRandom(random)
    public  void setRandom(boolean random)
    {
        
        setRandom_0(nativeObj, random);
        
        return;
    }


    //
    // C++:  void setSamples(int samples)
    //

    //javadoc: CalibrateDebevec::setSamples(samples)
    public  void setSamples(int samples)
    {
        
        setSamples_0(nativeObj, samples);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool getRandom()
    private static native boolean getRandom_0(long nativeObj);

    // C++:  float getLambda()
    private static native float getLambda_0(long nativeObj);

    // C++:  int getSamples()
    private static native int getSamples_0(long nativeObj);

    // C++:  void setLambda(float lambda)
    private static native void setLambda_0(long nativeObj, float lambda);

    // C++:  void setRandom(bool random)
    private static native void setRandom_0(long nativeObj, boolean random);

    // C++:  void setSamples(int samples)
    private static native void setSamples_0(long nativeObj, int samples);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
