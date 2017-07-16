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

        class Plot2dImpl : public Plot2d
        {
            public:

            Plot2dImpl(InputArray plotData)
            {
                Mat _plotData = plotData.getMat();
                //if the matrix is not Nx1 or 1xN
                if(_plotData.cols > 1 && _plotData.rows > 1)
                {
                    CV_Error(Error::StsBadArg, "ERROR: Plot data must be a 1xN or Nx1 matrix.\n");
                }

                //if the matrix type is not CV_64F
                if(_plotData.type() != CV_64F)
                {
                    CV_Error(Error::StsBadArg, "ERROR: Plot data type must be double (CV_64F).\n");
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

            Plot2dImpl(InputArray plotDataX_, InputArray plotDataY_)
            {
                Mat _plotDataX = plotDataX_.getMat();
                Mat _plotDataY = plotDataY_.getMat();
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
            }
            void setMaxX(double _plotMaxX)
            {
                plotMaxX = _plotMaxX;
            }
            void setMinY(double _plotMinY)
            {
                plotMinY = _plotMinY;
            }
            void setMaxY(double _plotMaxY)
            {
                plotMaxY = _plotMaxY;
            }
            void setPlotLineWidth(int _plotLineWidth)
            {
                plotLineWidth = _plotLineWidth;
            }
            void setNeedPlotLine(bool _needPlotLine)
            {
                needPlotLine = _needPlotLine;
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
            void render(InputOutputArray _plotResult)
            {
                bool renderAxes = false;
                if (_plotResult.empty())
                {
                    _plotResult.create(plotSizeHeight, plotSizeWidth, CV_8UC3);
                    _plotResult.setTo(plotBackgroundColor);
                    renderAxes = true; // only if this was a new image
                }

                plotResult = _plotResult.getMat();
                plotSizeHeight = plotResult.rows;
                plotSizeWidth  = plotResult.cols;

                Mat InterpXdata = linearInterpolation(plotMinX, plotMaxX, 0, plotSizeWidth, plotDataX);
                Mat InterpYdata = linearInterpolation(plotMinY, plotMaxY, 0, plotSizeHeight, plotDataY);

                int ImageXzero = int(-(plotMinX) * (plotSizeWidth  / (plotMaxX-plotMinX)));;
                int ImageYzero = int(-(plotMinY) * (plotSizeHeight / (plotMaxY-plotMinY)));;

                if (renderAxes)
                    drawAxes(ImageXzero, plotResult.rows - ImageYzero, plotAxisColor, plotGridColor);

                if (needPlotLine)
                {
                    //Draw the plot by connecting lines between the points
                    Point p1;
                    p1.x = (int)InterpXdata.at<double>(0,0);
                    p1.y = (int)InterpYdata.at<double>(0,0);
                    p1.y = plotSizeHeight - p1.y;

                    for (int r=1; r<InterpXdata.rows; r++)
                    {
                        Point p2;
                        p2.x = (int)InterpXdata.at<double>(r,0);
                        p2.y = (int)InterpYdata.at<double>(r,0);
                        p2.y = plotSizeHeight - p2.y;

                        line(plotResult, p1, p2, plotLineColor, plotLineWidth, 8, 0);

                        p1 = p2;
                    }
                }
                else
                {
                    for (int r=0; r<InterpXdata.rows; r++)
                    {
                        Point p;
                        p.x = (int)InterpXdata.at<double>(r,0);
                        p.y = (int)InterpYdata.at<double>(r,0);
                        p.y = plotSizeHeight - p.y;

                        circle(plotResult, p, 1, plotLineColor, plotLineWidth, 8, 0);
                    }
                }
            }

            protected:

            Mat plotDataX;
            Mat plotDataY;
            const char * plotName;

            //dimensions and limits of the plot
            int plotSizeWidth;
            int plotSizeHeight;
            double plotMinX;
            double plotMaxX;
            double plotMinY;
            double plotMaxY;
            int plotLineWidth;

            //colors of each plot element
            Scalar plotLineColor;
            Scalar plotBackgroundColor;
            Scalar plotAxisColor;
            Scalar plotGridColor;
            Scalar plotTextColor;

            //the final plot result
            Mat plotResult;

            //flag which enables/disables connection of plotted points by lines
            bool needPlotLine;

            void plotHelper(Mat _plotDataX, Mat _plotDataY)
            {
                plotDataX=_plotDataX;
                plotDataY=_plotDataY;

                needPlotLine = true;

                //setting the min and max values for each axis
                // (from input data), can be overridden later by setMaxX(...) etc.
                minMaxLoc(plotDataX, &plotMinX, &plotMaxX, 0, 0);
                minMaxLoc(plotDataY, &plotMinY, &plotMaxY, 0, 0);

                //setting the default size of a plot figure
                setPlotSize(600, 400);

                //setting the default plot line size
                setPlotLineWidth(1);

                //setting default colors for the different elements of the plot
                setPlotAxisColor(Scalar(0, 0, 255));
                setPlotGridColor(Scalar(105, 105, 105));
                setPlotBackgroundColor(Scalar(0, 0, 0));
                setPlotLineColor(Scalar(0, 255, 255));
                setPlotTextColor(Scalar(255, 255, 255));
            }

            void stipple(bool horz, int ImageXzero, int ImageYzero, int i, Scalar gridColor, int TraceSize = 5)
            {
                int mx = horz ? plotSizeWidth : plotSizeHeight;
                int Trace=0;
                while(Trace < mx)
                {
                    if (horz)
                        drawLine(Trace, Trace+TraceSize, ImageYzero+i, ImageYzero+i, gridColor);
                    else
                        drawLine(ImageXzero+i, ImageXzero+i, Trace, Trace+TraceSize, gridColor);
                    Trace = Trace + 2 * TraceSize;
                }
            }

            // draw a stippled line, and it's 0.5 resized (that's just an offset) downtoned version as well
            void stipple2(bool horz, int ImageXzero, int ImageYzero, int i1, int i2, Scalar gridColor)
            {
                stipple(horz, ImageXzero, ImageYzero, i1, gridColor);
                stipple(horz, ImageXzero, ImageYzero, i2, gridColor*0.5);
            }

            // find best grid size on pow 10 scale
            int bestGrid(double w, double instance)
            {
                double tile = w / instance;
                while(tile>w/5 ) { tile /= 10; }
                while(tile<w/10) { tile *= 10; }
                return (int)tile;
            }

            void drawAxes(int ImageXzero, int ImageYzero, Scalar axisColor, Scalar gridColor)
            {
                // if our actual zero value is "offside the grid", snap axis to border of the image
                int ImageXaxis = min(max(1,ImageXzero),plotSizeWidth-2);
                int ImageYaxis = min(max(1,ImageYzero),plotSizeHeight-2);
                // be prepared, that all neg y values we will need to draw on bottom side of x axis
                int dy = plotMaxY < 0 ? 25 : -5;
                drawValuesAsText(plotMaxX, plotSizeWidth - 30, ImageYaxis + dy);
                drawValuesAsText(plotMinX, 20, ImageYaxis + dy);
                // and left to the y axis, if all x are negative
                int dx = plotMaxX < 0 ? -25 : 5;
                drawValuesAsText(plotMaxY, ImageXaxis + dx, 10);
                drawValuesAsText(plotMinY, ImageXaxis + dx, plotSizeHeight - 20);

                //Horizontal X axis and equispaced horizontal lines
                drawLine(0, plotSizeWidth, ImageYaxis, ImageYaxis, axisColor);
                int LineSpace = bestGrid(plotSizeHeight, abs(plotMaxY - plotMinY));
                int maxIter = plotSizeHeight + abs(ImageYzero) + LineSpace;
                for (int i=-LineSpace; i>-maxIter; i=i-LineSpace)
                    stipple2(true, ImageXzero, ImageYzero, i, i+LineSpace/2, gridColor);
                for (int i=LineSpace; i<maxIter; i=i+LineSpace)
                    stipple2(true, ImageXzero, ImageYzero, i, i-LineSpace/2, gridColor);

                //Vertical Y axis
                drawLine(ImageXaxis, ImageXaxis, 0, plotSizeHeight, axisColor);
                LineSpace = bestGrid(plotSizeWidth, abs(plotMaxX - plotMinX));
                maxIter = plotSizeWidth + abs(ImageXzero) + LineSpace;
                for (int i=-LineSpace; i>-maxIter; i=i-LineSpace)
                    stipple2(false, ImageXzero, ImageYzero, i, i+LineSpace/2, gridColor);
                for (int i=LineSpace; i<maxIter; i=i+LineSpace)
                    stipple2(false, ImageXzero, ImageYzero, i, i-LineSpace/2, gridColor);
            }

            Mat linearInterpolation(double Xa, double Xb, double Ya, double Yb, const Mat &Xdata)
            {
                Mat Ydata = Xdata*0;
                for (int i=0; i<Xdata.rows; i++)
                {
                    double X = Xdata.at<double>(i,0);
                    Ydata.at<double>(i,0) = int(Ya + (Yb-Ya)*(X-Xa)/(Xb-Xa));

                    if(Ydata.at<double>(i,0)<0)
                        Ydata.at<double>(i,0)=0;
                }
                return Ydata;
            }

            void drawValuesAsText(double value, int Xloc, int Yloc)
            {
                double v = abs(value);
                const char *f = "%.1f";
                if (v>0 && v<0.5) f = "%.3f";
                if (v>0 && v<0.005) f = "%.5f";
                putText(plotResult, format(f,value), Point(Xloc,Yloc), FONT_HERSHEY_PLAIN, 0.85, plotTextColor, 1, 8);
            }

            void drawLine(int Xstart, int Xend, int Ystart, int Yend, Scalar lineColor)
            {
                line(plotResult, Point(Xstart,Ystart), Point(Xend,Yend), lineColor, plotLineWidth, 8, 0);
            }

        }; // cv::plot::Plot2dImpl

        Ptr<Plot2d> createPlot2d(InputArray _plotData)
        {
            return Ptr<Plot2dImpl> (new Plot2dImpl (_plotData));
        }

        Ptr<Plot2d> createPlot2d(InputArray _plotDataX, InputArray _plotDataY)
        {
            return Ptr<Plot2dImpl> (new Plot2dImpl (_plotDataX, _plotDataY));
        }

    }
}
