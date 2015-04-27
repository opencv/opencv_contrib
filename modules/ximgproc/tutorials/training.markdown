Structured forest training {#tutorial_ximgproc_training}
==========================

Introduction
------------

In this tutorial we show how to train your own structured forest using author's initial Matlab
implementation.

Training pipeline
-----------------

-#  Download "Piotr's Toolbox" from [link](http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html)
    and put it into separate directory, e.g. PToolbox

-#  Download BSDS500 dataset from
    link \<http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/\> and put it into
    separate directory named exactly BSR
-#  Add both directory and their subdirectories to Matlab path.

-#  Download detector code from
    link \<http://research.microsoft.com/en-us/downloads/389109f6-b4e8-404c-84bf-239f7cbf4e3d/\> and
    put it into root directory. Now you should have :
    @code
        .
            BSR
            PToolbox
            models
            private
            Contents.m
            edgesChns.m
            edgesDemo.m
            edgesDemoRgbd.m
            edgesDetect.m
            edgesEval.m
            edgesEvalDir.m
            edgesEvalImg.m
            edgesEvalPlot.m
            edgesSweeps.m
            edgesTrain.m
            license.txt
            readme.txt
    @endcode

-#  Rename models/forest/modelFinal.mat to models/forest/modelFinal.mat.backup

-#  Open edgesChns.m and comment lines 26--41. Add after commented lines the following:
    @code{.cpp}
            shrink=opts.shrink;
            chns = single(getFeatures( im2double(I) ));
    @endcode

-#  Now it is time to compile promised getFeatures. I do with the following code:
    @code{.cpp}
    #include <cv.h>
    #include <highgui.h>

    #include <mat.h>
    #include <mex.h>

    #include "MxArray.hpp" // https://github.com/kyamagu/mexopencv

    class NewRFFeatureGetter : public cv::RFFeatureGetter
    {
    public:
        NewRFFeatureGetter() : name("NewRFFeatureGetter"){}

        virtual void getFeatures(const cv::Mat &src, NChannelsMat &features,
                                 const int gnrmRad, const int gsmthRad,
                                 const int shrink, const int outNum, const int gradNum) const
        {
            // here your feature extraction code, the default one is:
            // resulting features Mat should be n-channels, floating point matrix
        }

    protected:
        cv::String name;
    };

    MEXFUNCTION_LINKAGE void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
    {
        if (nlhs != 1) mexErrMsgTxt("nlhs != 1");
        if (nrhs != 1) mexErrMsgTxt("nrhs != 1");

        cv::Mat src = MxArray(prhs[0]).toMat();
        src.convertTo(src, cv::DataType<float>::type);

        std::string modelFile = MxArray(prhs[1]).toString();
        NewRFFeatureGetter *pDollar = createNewRFFeatureGetter();

        cv::Mat edges;
        pDollar->getFeatures(src, edges, 4, 0, 2, 13, 4);
        // you can use other numbers here

        edges.convertTo(edges, cv::DataType<double>::type);

        plhs[0] = MxArray(edges);
    }
    @endcode

-#  Place compiled mex file into root dir and run edgesDemo. You will need to wait a couple of hours
    after that the new model will appear inside models/forest/.

-#  The final step is converting trained model from Matlab binary format to YAML which you can use
    with our ocv::StructuredEdgeDetection. For this purpose run
    opencv_contrib/ximpgroc/tutorials/scripts/modelConvert(model, "model.yml")

How to use your model
---------------------

Just use expanded constructor with above defined class NewRFFeatureGetter
@code{.cpp}
cv::StructuredEdgeDetection pDollar
    = cv::createStructuredEdgeDetection( modelName, makePtr<NewRFFeatureGetter>() );
@endcode
