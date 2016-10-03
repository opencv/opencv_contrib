
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.photo;



// C++: class TonemapDrago
//javadoc: TonemapDrago
public class TonemapDrago extends Tonemap {

    protected TonemapDrago(long addr) { super(addr); }


    //
    // C++:  float getBias()
    //

    //javadoc: TonemapDrago::getBias()
    public  float getBias()
    {
        
        float retVal = getBias_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getSaturation()
    //

    //javadoc: TonemapDrago::getSaturation()
    public  float getSaturation()
    {
        
        float retVal = getSaturation_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setBias(float bias)
    //

    //javadoc: TonemapDrago::setBias(bias)
    public  void setBias(float bias)
    {
        
        setBias_0(nativeObj, bias);
        
        return;
    }


    //
    // C++:  void setSaturation(float saturation)
    //

    //javadoc: TonemapDrago::setSaturation(saturation)
    public  void setSaturation(float saturation)
    {
        
        setSaturation_0(nativeObj, saturation);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  float getBias()
    private static native float getBias_0(long nativeObj);

    // C++:  float getSaturation()
    private static native float getSaturation_0(long nativeObj);

    // C++:  void setBias(float bias)
    private static native void setBias_0(long nativeObj, float bias);

    // C++:  void setSaturation(float saturation)
    private static native void setSaturation_0(long nativeObj, float saturation);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
