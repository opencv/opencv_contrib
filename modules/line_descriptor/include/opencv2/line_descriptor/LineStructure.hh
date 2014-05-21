/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2011-2012, Lilian Zhang, all rights reserved.
Copyright (C) 2013, Manuele Tamburrano, Stefano Fabri, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef LINESTRUCTURE_HH_
#define LINESTRUCTURE_HH_

#include <vector>
// A 2D line (normal equation parameters).
struct SingleLine
{
	//note: rho and theta are based on coordinate origin, i.e. the top-left corner of image
	double rho;//unit: pixel length
	double theta;//unit: rad
	double linePointX;// = rho * cos(theta);
	double linePointY;// = rho * sin(theta);
	//for EndPoints, the coordinate origin is the top-left corner of image.
	double startPointX;
	double startPointY;
	double endPointX;
	double endPointY;
	//direction of a line, the angle between positive line direction (dark side is in the left) and positive X axis.
	double direction;
	//mean gradient magnitude
	double gradientMagnitude;
	//mean gray value of pixels in dark side of line
	double darkSideGrayValue;
	//mean gray value of pixels in light side of line
	double lightSideGrayValue;
	//the length of line
	double lineLength;
	//the width of line;
	double width;
	//number of pixels
	int numOfPixels;
	//the decriptor of line
	std::vector<double> descriptor;
};

// Specifies a vector of lines.
typedef std::vector<SingleLine> Lines_list;

struct OctaveSingleLine
{
	/*endPoints, the coordinate origin is the top-left corner of the original image.
	 *startPointX = sPointInOctaveX * (factor)^octaveCount;	*/
	float startPointX;
	float startPointY;
	float endPointX;
	float endPointY;
	//endPoints, the coordinate origin is the top-left corner of the octave image.
	float sPointInOctaveX;
	float sPointInOctaveY;
	float ePointInOctaveX;
	float ePointInOctaveY;
	//direction of a line, the angle between positive line direction (dark side is in the left) and positive X axis.
	float direction;
	//the summation of gradient magnitudes of pixels on lines
	float salience;
	//the length of line
	float lineLength;
	//number of pixels
	unsigned int numOfPixels;
	//the octave which this line is detected
	unsigned int octaveCount;
	//the decriptor of line
	std::vector<float> descriptor;
};

// Specifies a vector of lines.
typedef std::vector<OctaveSingleLine> LinesVec;

typedef std::vector<LinesVec> ScaleLines;//each element in ScaleLines is a vector of lines which corresponds the same line detected in different octave images.

#endif /* LINESTRUCTURE_HH_ */
