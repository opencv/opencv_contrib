/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
//################################################################################
//
//                    Created by Nuno Moutinho
//
//################################################################################

#include "precomp.hpp"

namespace cv
{
    namespace plot
    {
        using namespace std;

        class Plot2dImpl : public Plot2d
        {
            public:

            Plot2dImpl(Mat _plotData)
            {
                //if the matrix is not Nx1 or 1xN
                if(_plotData.cols > 1 && _plotData.rows > 1)
                {
                    std::cout << "ERROR: Plot data must be a 1xN or Nx1 matrix." << std::endl;
                    exit(0);
                }

                //if the matrix type is not CV_64F
                if(_plotData.type() != CV_64F)
                {
                    std::cout << "ERROR: Plot data type must be double (CV_64F)." << std::endl;
                    exit(0);
                }

                //in case we have a row matrix than needs to be transposed
                if(_plotData.cols > _plotData.rows)
                {
                    _plotData = _plotData.t();
                }

                plotDataY=_plotData;
                plotDataX = plotDataY*0;
                for (int i=0; i<plotDataY.rows; i++)
                {
                    plotDataX.at<double>(i,0) = i;
                }

                //calling the main constructor
                plotHelper(plotDataX, plotDataY);

            }

            Plot2dImpl(Mat _plotDataX, Mat _plotDataY)
            {
                //f the matrix is not Nx1 or 1xN
                if((_plotDataX.cols > 1 && _plotDataX.rows > 1) || (_plotDataY.cols > 1 && _plotDataY.rows > 1))
                {
                    std::cout << "ERROR: Plot data must be a 1xN or Nx1 matrix." << std::endl;
                    exit(0);
                }

                //if the matrix type is not CV_64F
                if(_plotDataX.type() != CV_64F || _plotDataY.type() != CV_64F)
                {
                    std::cout << "ERROR: Plot data type must be double (CV_64F)." << std::endl;
                   exit(0);
                }

                //in case we have a row matrix than needs to be transposed
                if(_plotDataX.cols > _plotDataX.rows)
                {
                    _plotDataX = _plotDataX.t();
                }
                if(_plotDataY.cols > _plotDataY.rows)
                {
                    _plotDataY = _plotDataY.t();
                }

                plotHelper(_plotDataX, _plotDataY);
            }

            //set functions
            void setMinX(double _plotMinX)
            {
                plotMinX = _plotMinX;
                plotMinX_plusZero = _plotMinX;
            }
            void setMaxX(double _plotMaxX)
            {
                plotMaxX = _plotMaxX;
                plotMaxX_plusZero = _plotMaxX;
            }
            void setMinY(double _plotMinY)
            {
                plotMinY = _plotMinY;
                plotMinY_plusZero = _plotMinY;
            }
            void setMaxY(double _plotMaxY)
            {
                plotMaxY = _plotMaxY;
                plotMaxY_plusZero = _plotMaxY;
            }
            void setPlotLineWidth(int _plotLineWidth)
            {
                plotLineWidth=_plotLineWidth;
            }
            void setPlotLineColor(Scalar _plotLineColor)
            {
                plotLineColor=_plotLineColor;
            }
            void setPlotBackgroundColor(Scalar _plotBackgroundColor)
            {
                plotBackgroundColor=_plotBackgroundColor;
            }
            void setPlotAxisColor(Scalar _plotAxisColor)
            {
                plotAxisColor=_plotAxisColor;
            }
            void setPlotGridColor(Scalar _plotGridColor)
            {
                plotGridColor=_plotGridColor;
            }
            void setPlotTextColor(Scalar _plotTextColor)
            {
                plotTextColor=_plotTextColor;
            }
            void setPlotSize(int _plotSizeWidth, int _plotSizeHeight)
            {
                if(_plotSizeWidth > 400)
                    plotSizeWidth = _plotSizeWidth;
                else
                    plotSizeWidth = 400;

                if(_plotSizeHeight > 300)
                    plotSizeHeight = _plotSizeHeight;
                else
                    plotSizeHeight = 300;
            }

            //render the plotResult to a Mat
            void render(Mat &_plotResult)
            {
                //create the plot result
                plotResult = Mat(plotSizeHeight, plotSizeWidth, CV_8UC3, plotBackgroundColor);

                int NumVecElements = plotDataX.rows;

                Mat InterpXdata = linearInterpolation(plotMinX, plotMaxX, 0, plotSizeWidth, plotDataX);
                Mat InterpYdata = linearInterpolation(plotMinY, plotMaxY, 0, plotSizeHeight, plotDataY);

                //Find the zeros in image coordinates
                Mat InterpXdataFindZero = linearInterpolation(plotMinX_plusZero, plotMaxX_plusZero, 0, plotSizeWidth, plotDataX_plusZero);
                Mat InterpYdataFindZero = linearInterpolation(plotMinY_plusZero, plotMaxY_plusZero, 0, plotSizeHeight, plotDataY_plusZero);

                int ImageXzero = (int)InterpXdataFindZero.at<double>(NumVecElements,0);
                int ImageYzero = (int)InterpYdataFindZero.at<double>(NumVecElements,0);

                double CurrentX = plotDataX.at<double>(NumVecElements-1,0);
                double CurrentY = plotDataY.at<double>(NumVecElements-1,0);

                //Draw the plot by connecting lines between the points
                Point p1;
                p1.x = (int)InterpXdata.at<double>(0,0);
                p1.y = (int)InterpYdata.at<double>(0,0);

                drawAxis(ImageXzero,ImageYzero, CurrentX, CurrentY, plotAxisColor, plotGridColor);

                for (int r=1; r<InterpXdata.rows; r++){

                    Point p2;
                    p2.x = (int)InterpXdata.at<double>(r,0);
                    p2.y = (int)InterpYdata.at<double>(r,0);

                    line(plotResult, p1, p2, plotLineColor, plotLineWidth, 8, 0);

                    p1 = p2;

                }

                _plotResult = plotResult.clone();
            }

            protected:

            Mat plotDataX;
            Mat plotDataY;
            Mat plotDataX_plusZero;
            Mat plotDataY_plusZero;
            const char * plotName;

            //dimensions and limits of the plot
            int plotSizeWidth;
            int plotSizeHeight;
            double plotMinX;
            double plotMaxX;
            double plotMinY;
            double plotMaxY;
            double plotMinX_plusZero;
            double plotMaxX_plusZero;
            double plotMinY_plusZero;
            double plotMaxY_plusZero;
            int plotLineWidth;

            //colors of each plot element
            Scalar plotLineColor;
            Scalar plotBackgroundColor;
            Scalar plotAxisColor;
            Scalar plotGridColor;
            Scalar plotTextColor;

            //the final plot result
            Mat plotResult;

            void plotHelper(Mat _plotDataX, Mat _plotDataY)
            {
                plotDataX=_plotDataX;
                plotDataY=_plotDataY;

                int NumVecElements = plotDataX.rows;

                plotDataX_plusZero = Mat::zeros(NumVecElements+1,1,CV_64F);
                plotDataY_plusZero = Mat::zeros(NumVecElements+1,1,CV_64F);

                for(int i=0; i<NumVecElements; i++){
                    plotDataX_plusZero.at<double>(i,0) = plotDataX.at<double>(i,0);
                    plotDataY_plusZero.at<double>(i,0) = plotDataY.at<double>(i,0);
                }

                double MinX;
                double MaxX;
                double MinY;
                double MaxY;
                double MinX_plusZero;
                double MaxX_plusZero;
                double MinY_plusZero;
                double MaxY_plusZero;

                //Obtain the minimum and maximum values of Xdata
                minMaxLoc(plotDataX,&MinX,&MaxX);

                //Obtain the minimum and maximum values of Ydata
                minMaxLoc(plotDataY,&MinY,&MaxY);

                //Obtain the minimum and maximum values of Xdata plus zero
                minMaxLoc(plotDataX_plusZero,&MinX_plusZero,&MaxX_plusZero);

                //Obtain the minimum and maximum values of Ydata plus zero
                minMaxLoc(plotDataY_plusZero,&MinY_plusZero,&MaxY_plusZero);

                //setting the min and max values for each axis
                plotMinX = MinX;
                plotMaxX = MaxX;
                plotMinY = MinY;
                plotMaxY = MaxY;
                plotMinX_plusZero = MinX_plusZero;
                plotMaxX_plusZero = MaxX_plusZero;
                plotMinY_plusZero = MinY_plusZero;
                plotMaxY_plusZero = MaxY_plusZero;

                //setting the default size of a plot figure
                setPlotSize(600, 400);

                //setting the default plot line size
                setPlotLineWidth(1);

                //setting default colors for the different elements of the plot
                setPlotAxisColor(Scalar(0, 0, 255));
                setPlotGridColor(Scalar(255, 255, 255));
                setPlotBackgroundColor(Scalar(0, 0, 0));
                setPlotLineColor(Scalar(0, 255, 255));
                setPlotTextColor(Scalar(255, 255, 255));
            }

            void drawAxis(int ImageXzero, int ImageYzero, double CurrentX, double CurrentY, Scalar axisColor, Scalar gridColor)
            {
                drawValuesAsText(0, ImageXzero, ImageYzero, 10, 20);
                drawValuesAsText(0, ImageXzero, ImageYzero, -20, 20);
                drawValuesAsText(0, ImageXzero, ImageYzero, 10, -10);
                drawValuesAsText(0, ImageXzero, ImageYzero, -20, -10);
                drawValuesAsText("X = %g",CurrentX, 0, 0, 40, 20);
                drawValuesAsText("Y = %g",CurrentY, 0, 20, 40, 20);

                //Horizontal X axis and equispaced horizontal lines
                int LineSpace = 50;
                int TraceSize = 5;
                drawLine(0, plotSizeWidth, ImageYzero, ImageYzero, axisColor);

               for(int i=-plotSizeHeight; i<plotSizeHeight; i=i+LineSpace){

                    if(i!=0){
                        int Trace=0;
                        while(Trace<plotSizeWidth){
                            drawLine(Trace, Trace+TraceSize, ImageYzero+i, ImageYzero+i, gridColor);
                            Trace = Trace+2*TraceSize;
                        }
                    }
                }


                //Vertical Y axis
                drawLine(ImageXzero, ImageXzero, 0, plotSizeHeight, axisColor);

                for(int i=-plotSizeWidth; i<plotSizeWidth; i=i+LineSpace){

                    if(i!=0){
                        int Trace=0;
                        while(Trace<plotSizeHeight){
                            drawLine(ImageXzero+i, ImageXzero+i, Trace, Trace+TraceSize, gridColor);
                            Trace = Trace+2*TraceSize;
                        }
                    }
                }
            }

            Mat linearInterpolation(double Xa, double Xb, double Ya, double Yb, Mat Xdata){

                Mat Ydata = Xdata*0;

                for (int i=0; i<Xdata.rows; i++){

                    double X = Xdata.at<double>(i,0);
                    Ydata.at<double>(i,0) = int(Ya + (Yb-Ya)*(X-Xa)/(Xb-Xa));

                    if(Ydata.at<double>(i,0)<0)
                        Ydata.at<double>(i,0)=0;

                }

                return Ydata;
            }

            void drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin){

                char AxisX_Min_Text[20];
                double TextSize = 1;

                sprintf(AxisX_Min_Text, "%g", Value);
                Point AxisX_Min_Loc;
                AxisX_Min_Loc.x = Xloc+XMargin;
                AxisX_Min_Loc.y = Yloc+YMargin;

                putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);
            }

            void drawValuesAsText(const char *Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin){

                char AxisX_Min_Text[20];
                int TextSize = 1;

                sprintf(AxisX_Min_Text, Text, Value);
                Point AxisX_Min_Loc;
                AxisX_Min_Loc.x = Xloc+XMargin;
                AxisX_Min_Loc.y = Yloc+YMargin;

                putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);
            }


            void drawLine(int Xstart, int Xend, int Ystart, int Yend, Scalar lineColor){

                Point Axis_start;
                Point Axis_end;
                Axis_start.x = Xstart;
                Axis_start.y = Ystart;
                Axis_end.x = Xend;
                Axis_end.y = Yend;

                line(plotResult, Axis_start, Axis_end, lineColor, plotLineWidth, 8, 0);
            }

        };

        Ptr<Plot2d> createPlot2d(Mat _plotData)
        {
            return Ptr<Plot2dImpl> (new Plot2dImpl (_plotData));

        }

        Ptr<Plot2d> createPlot2d(Mat _plotDataX, Mat _plotDataY)
        {
            return Ptr<Plot2dImpl> (new Plot2dImpl (_plotDataX, _plotDataY));
        }
    }
}
