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

#ifndef __OPENCV_ADAS_ADAS_HPP__
#define __OPENCV_ADAS_ADAS_HPP__

namespace cv
{
namespace adas
{

class CV_EXPORTS ICFDetector
{
public:
    /* Initialize detector */
    ICFDetector();

    /* Load detector from file, return true on success, false otherwise */
    bool load(const std::string& filename);

    /* Run detector on single image

        image — image for detection

        bboxes — output array of object bounding boxes in format
                 Rect(row_from, col_from, n_rows, n_cols)

        confidence_values — output array of confidence values from 0 to 1.
            One value per bbox — confidence of detector that corresponding
            bbox contatins object

        minObjSize — min possible object size on image in pixels (rows x cols)

        maxObjSize — max possible object size on image in pixels (rows x cols)

    */
    void detect(InputArray image,
                std::vector<Rect>& bboxes,
                std::vector<float>& confidence_values,
                Size minObjSize,
                Size maxObjSize) const;
};

} /* namespace adas */
} /* namespace cv */

#endif /* __OPENCV_ADAS_ADAS_HPP__ */
