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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __DPM_MODEL__
#define __DPM_MODEL__

#include "opencv2/core.hpp"

#include <string>
#include <vector>

namespace cv
{
namespace dpm
{
/** @brief This class contains DPM model parameters
 */
class Model
{
    public:
        // size of HOG feature cell (e.g., 8 pixels)
        int sBin;
        // number of levels per octave in feature pyramid
        int interval;
        // maximum width of the detection window
        int maxSizeX;
        // maximum height of the detection window
        int maxSizeY;
        // dimension of HOG features
        int numFeatures;
        // number of components in the model
        int numComponents;
        // number of parts per component
        std::vector<int> numParts;
        // size of root filters
        std::vector< Size > rootFilterDims;
        // size of part filters
        std::vector< Size > partFilterDims;
        // root filters
        std::vector< Mat > rootFilters;
        // part filters
        std::vector< Mat > partFilters;
        // global detecion threshold
        float scoreThresh;
        // component indexed array of part orderings
        std::vector< std::vector<int> > partOrder;
        // component indexed offset (a.k.a. bias) values
        std::vector<float> bias;
        // location/scale weight
        std::vector< std::vector< double > >  locationWeight;
        // idea relative positions for each deformation model
        std::vector< std::vector< double > >  anchors;
        // array of deformation models
        std::vector< std::vector< double > > defs;

        // map: pFind[component][part] => part filter index
        std::vector< std::vector<int> > pFind;

    public:
        Model () {}
        virtual ~Model () {}

        // get number of part filters
        int getNumPartFilters()
        {
            return (int) partFilters.size();
        }

        // get number of deformation parameters
        int getNumDefParams()
        {
            return (int) defs.size();
        }

        virtual void initModel() {};
        virtual bool serialize(const std::string &filename) const = 0;
        virtual bool deserialize(const std::string &filename) = 0;
};

class CascadeModel : public Model
{
    public:
        // PCA coefficient matrix
        Mat pcaCoeff;
        // number of dimensions used for the PCA projection
        int pcaDim;
        // component indexed arrays of pruning threshold
        std::vector< std::vector< double > >  prunThreshold;
        // root pca filters
        std::vector< Mat > rootPCAFilters;
        // part PCA filters
        std::vector< Mat > partPCAFilters;
    public:
        CascadeModel() {}
        ~CascadeModel() {}

        void initModel();
        bool serialize(const std::string &filename) const;
        bool deserialize(const std::string &filename);
};

} // namespace lsvm
} // namespace cv

#endif // __DPM_MODEL_
