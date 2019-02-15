#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <iostream>

using namespace cv;

int main()
{
    Mat data_x( 1, 51, CV_64F );
    Mat data_y( 1, 51, CV_64F );

    for ( int i = 0; i < data_x.cols; i++ )
    {
        double x = ( i - data_x.cols / 2 );
        data_x.at<double>( 0, i ) = x;
        data_y.at<double>( 0, i ) = x * x * x;
    }

    std::cout << "data_x : " << data_x << std::endl;
    std::cout << "data_y : " << data_y << std::endl;

    Mat plot_result;

    Ptr<plot::Plot2d> plot = plot::Plot2d::create( data_x, data_y );
    plot->render(plot_result);

    imshow( "The plot rendered with default visualization options", plot_result );

    plot->setShowText( false );
    plot->setShowGrid( false );
    plot->setPlotBackgroundColor( Scalar( 255, 200, 200 ) );
    plot->setPlotLineColor( Scalar( 255, 0, 0 ) );
    plot->setPlotLineWidth( 2 );
    plot->setInvertOrientation( true );
    plot->render( plot_result );

    imshow( "The plot rendered with some of custom visualization options", plot_result );
    waitKey();

    return 0;
}
