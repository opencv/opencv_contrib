/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#ifndef __OPENCV_FACE_ALIGNMENT_HPP__
#define __OPENCV_FACE_ALIGNMENT_HPP__

#include <string>
#include <vector>


namespace cv{ namespace face {

class CV_EXPORTS_W FaceAlignment : public Algorithm
{
public:

/** @returns The number of landmarks detected in an image */
    CV_WRAP virtual int getNumLandmarks() const = 0;

    /** @returns The number of faces detected in an image */
    CV_WRAP virtual int getNumFaces() const = 0;


    CV_WRAP  virtual std::vector<cv::Point2f> getLandmarks(
            cv::Mat img,
            const cv::Rect face,
            std::vector<cv::Point2f> > initial_feats
        ) const=0;
    /** This function takes the initial image,one of the face and mean shape in the image 
    and returns the landmarks in the image. 
      The description of the arguments is as follows :
      @param img(Mat)- It recieves a Mat object in which it detects the landmarks.
      @param face(Rect)- It recieves one of the bounding rectangle of one of the face in the image.
      @param initial_feats(vector<Point2f>- It recieves initial position of landmarks which is there in the mean image.
          It is a vector of points which denote the position of landmarks in the image or the initial shape**/

};
/**
 * @param num_faces The number of faces (>=1) for computing shape.
 * @param num_landmarks The number of landmarks(=194) according to the HELEN dataset being used.
 * @returns Object for computing shape.
 */


CV_EXPORTS_W cv::Ptr<FaceAlignment> createFaceAlignment(int num_faces = 1, int num_landmarks = 194,cv::Mat img);

}//face
}//cv
#endif

