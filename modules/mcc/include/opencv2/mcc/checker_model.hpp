
#ifndef __OPENCV_MCC_CHECKER_MODEL_HPP__
#define __OPENCV_MCC_CHECKER_MODEL_HPP__
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
namespace cv
{
namespace mcc
{

//! @addtogroup mcc
//! @{

/** TYPECHART
 *
 * \brief enum to hold the type of the checker
 *
 */
enum TYPECHART
{
    MCC24=0,  ///< Standard Macbeth Chart with 24 squares
    SG140,  ///< DigitalSG with 140 squares
    VINYL18,///< DKK color chart with 12 squares and 6 rectangle

};

/** CChecker
 *
 * \brief checker object
 *
 * 	This class contains the information about the detected checkers,i.e, their
 * 	type, the corners of the chart, the color profile, the cost, centers chart,
 * 	etc.
 *
 */

class CV_EXPORTS_W CChecker
{
public:
	CChecker(){}
	~CChecker(){}
	/** \brief Create a new CChecker object.
	* \return A pointer to the implementation of the CChecker
	*/

	CV_WRAP static Ptr<CChecker> create();

public:
	CV_PROP_RW TYPECHART target;			  ///< type of checkercolor
	CV_PROP_RW std::vector<cv::Point2f> box; ///< positions of the corners
	CV_PROP_RW cv::Mat charts_rgb;			  ///< charts profile in rgb color space
	CV_PROP_RW cv::Mat charts_ycbcr;		  ///< charts profile in YCbCr color space
	CV_PROP_RW float cost;					  ///< cost to aproximate
	CV_PROP_RW cv::Point2f center;           ///< center of the chart.
};

/** \brief checker draw
 *
 *  This class contains the functions for drawing a detected chart.  This
 *  class expects a pointer to the checker which will be drawn by this
 *  object in the constructor and then later on whenever the draw function
 *  is called the checker will be drawn. Remember that it is not possible
 *  to change the checkers which will be draw by a given object, as it is
 *  decided in the constructor itself. If you want to draw some other
 *  object you can create a new CCheckerDraw instance.
 *
 *  The reason for this type of design is that in some videos we can
 *  assume that the checker is always in the same position, even if the
 *  image changes, so the drawing will always take place at the same position.
*/
class CV_EXPORTS_W CCheckerDraw
{

public:
	virtual ~CCheckerDraw() {}
	/** \brief Draws the checker to the given image.
	* \param img image in color space BGR
	* \return void
	*/
	CV_WRAP virtual void draw(cv::Mat &img) = 0;
	/** \brief Create a new CCheckerDraw object.
	* \param pChecker The checker which will be drawn by this object.
	* \param color The color by with which the squares of the checker
	*              will be drawn
	* \param thickness The thickness with which the sqaures will be
	*                  drawn
	* \return A pointer to the implementation of the CCheckerDraw
	*/
	CV_WRAP static Ptr<CCheckerDraw> create(Ptr<CChecker> pChecker, cv::Scalar color = CV_RGB(0, 250, 0), int thickness = 2);
};

//! @} mcc
} // namespace mcc
} // namespace cv

#endif
