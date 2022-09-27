/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"

namespace cv{
  namespace ximgproc{

    // Function prototype
    void letterboxResize(InputArray _src, OutputArray _dst, Size dSize, int interpolation, int borderType, Scalar value);

    // Function implementation
    void letterboxResize(InputArray _src, OutputArray _dst, Size dSize, int interpolation, int borderType, Scalar value)
    {

      double scale       = 0;
      int    dWidth      = 0;
      int    dHeight     = 0;
      int    sWidth      = 0;
      int    sHeight     = 0;
      int    newWidth    = 0;
      int    newHeight   = 0;
      int    deltaWidth  = 0;
      int    deltaHeight = 0;
      int    top         = 0;
      int    bottom      = 0;
      int    left        = 0;
      int    right       = 0;

      CV_Assert( !dSize.empty() );
      dWidth  = dSize.width;
      dHeight = dSize.height;

      Size ssize = _src.size();
      CV_Assert( !ssize.empty() );
      sWidth  = ssize.width;
      sHeight = ssize.height;

      // Calculate scale as minimum of width/matWidth and height/matHeight
      scale = (double) dWidth / sWidth;
      if (scale > (double) dHeight / sHeight)
      {
        scale = (double) dHeight / sHeight;
      }

      // Calculate new width and height using scale
      newWidth  = (int) (scale * sWidth);
      newHeight = (int) (scale * sHeight);

      Size rsize = Size(newWidth, newHeight);

      // Resize image to newWidth and newHeight
      Mat src = _src.getMat();

      cv::resize( src, src, rsize, 0, 0, interpolation );

      // Calculate border sizes
      deltaWidth  = dWidth - newWidth;
      deltaHeight = dHeight - newHeight;

      top    = floor(deltaHeight / 2);
      bottom = deltaHeight - top;

      left  = floor(deltaWidth / 2);
      right = deltaWidth - left;

      _dst.create(dSize, src.type());
      Mat dst = _dst.getMat();

      cv::copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );

    }
  }
}
