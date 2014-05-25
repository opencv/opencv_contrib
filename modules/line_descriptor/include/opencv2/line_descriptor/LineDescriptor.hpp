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

#ifndef LINEDESCRIPTOR_HH_
#define LINEDESCRIPTOR_HH_


//#include "EDLineDetector.hh"
//#include "LineStructure.hh"

//#include "/include/opencv2/line_descriptor/EDLineDetector.hpp"
//#include "opencv2/line_descriptor/LineStructure.hpp"

//#include "opencv2/EDLineDetector.hpp"
//#include "opencv2/LineStructure.hpp"

#include "precomp.hpp"

#include <vector>
#include <map>

struct OctaveLine{
  unsigned int octaveCount;//the octave which this line is detected
  unsigned int lineIDInOctave;//the line ID in that octave image
  unsigned int lineIDInScaleLineVec;//the line ID in Scale line vector
  float lineLength; //the length of line in original image scale
};


/* This class is used to generate the line descriptors from multi-scale images  */
class LineDescriptor
{
public:
	LineDescriptor(int n_octave);
	LineDescriptor(unsigned int numOfBand, unsigned int widthOfBand, int n_octave);
	~LineDescriptor();
	enum{
		NearestNeighbor=0, //the nearest neighbor is taken as matching
		NNDR=1//nearest/next ratio
	};

  /*This function is used to detect lines from multi-scale images.*/
  int OctaveKeyLines(cv::Mat & image, ScaleLines &keyLines);
  int GetLineDescriptor(cv::Mat & image, ScaleLines &keyLines);
  int GetLineDescriptor(std::vector<cv::Mat> &scale_images, ScaleLines &keyLines);

  void findNearestParallelLines(ScaleLines & keyLines);

  void GetLineBinaryDescriptor(std::vector<cv::Mat> & oct_binaryDescMat, ScaleLines &keyLines);
  int MatchLineByDescriptor(ScaleLines &keyLinesLeft, ScaleLines &keyLinesRight,
  		std::vector<short> &matchLeft, std::vector<short> &matchRight,
  		int criteria=NNDR);
  float LowestThreshold;//global threshold for line descriptor distance, default is 0.35
  float NNDRThreshold;//the NNDR threshold for line descriptor distance, default is 0.6
  
  int ksize_; //the size of Gaussian kernel: ksize X ksize, default value is 5.
  unsigned int  numOfOctave_;//the number of image octave
  unsigned int  numOfBand_;//the number of band used to compute line descriptor
  unsigned int  widthOfBand_;//the width of band;
  std::vector<float> gaussCoefL_;//the local gaussian coefficient apply to the orthogonal line direction within each band;
  std::vector<float> gaussCoefG_;//the global gaussian coefficient apply to each Row within line support region
  
  
private:

	void sample(float *igray,float *ogray, float factor, int width, int height)
	{

		int swidth = (int)((float) width / factor);
		int sheight = (int)((float) height / factor);

		for(int j=0; j < sheight; j++)
		 for(int i=0; i < swidth; i++)
			ogray[j*swidth + i] = igray[(int)((float) j * factor) * width + (int) ((float) i*factor)];

	}
	void sampleUchar(uchar *igray,uchar *ogray, float factor, int width, int height)
    {

        int swidth = (int)((float) width / factor);
        int sheight = (int)((float) height / factor);

        for(int j=0; j < sheight; j++)
         for(int i=0; i < swidth; i++)
            ogray[j*swidth + i] = igray[(int)((float) j * factor) * width + (int) ((float) i*factor)];

    }
	/*Compute the line descriptor of input line set. This function should be called
	 *after OctaveKeyLines() function; */
	int ComputeLBD_(ScaleLines &keyLines);
	/*For each octave of image, we define an EDLineDetector, because we can get gradient images (dxImg, dyImg, gImg)
	 *from the EDLineDetector class without extra computation cost. Another reason is that, if we use
	 *a single EDLineDetector to detect lines in different octave of images, then we need to allocate and release
	 *memory for gradient images (dxImg, dyImg, gImg) repeatedly for their varying size*/
    std::vector<EDLineDetector*> edLineVec_;

	
};






#endif /* LINEDESCRIPTOR_HH_ */
