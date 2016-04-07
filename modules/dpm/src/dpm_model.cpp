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

#include "dpm_model.hpp"

namespace cv
{
namespace dpm
{

void CascadeModel::initModel()
{
    CV_Assert(numComponents == (int)rootFilters.size());
    // map: pFind[component][part] => part filter index
    pFind.resize(numComponents);
    int np = (int) partFilters.size();
    rootFilterDims.resize(numComponents);
    partFilterDims.resize(np);

    int w, h; // width and height of the filter
    int pIndex = 0; // part index

    for (int comp  = 0; comp < numComponents; comp++)
    {
        w = rootFilters[comp].cols/numFeatures;
        h = rootFilters[comp].rows;
        rootFilterDims[comp] = Size(w, h);
        pFind[comp].resize(numParts[comp]);

        for (int part = 0; part < numParts[comp]; part++)
        {
            w = partFilters[pIndex].cols/numFeatures;
            h = partFilters[pIndex].rows;
            partFilterDims[pIndex] = Size(w, h);
            pFind[comp][part] = pIndex;
            pIndex++;
        }
    }

    CV_Assert(pIndex == np);
    CV_Assert(pIndex == (int)anchors.size());
    CV_Assert(pIndex == (int)defs.size());
}

bool CascadeModel::serialize(const std::string &filename) const
{
    // open the storage container for writing
    FileStorage fs;
    fs.open(filename, FileStorage::WRITE);

    // write the primitives
    fs << "SBin" << sBin;
    fs << "Interval" << interval;
    fs << "MaxSizeX" << maxSizeX;
    fs << "MaxSizeY" << maxSizeY;
    fs << "NumComponents" << numComponents;
    fs << "NumFeatures" << numFeatures;
    fs << "PCADim" << pcaDim;
    fs << "ScoreThreshold" << scoreThresh;
    fs << "PCAcoeff" << pcaCoeff;
    fs << "Bias" << bias;

    // write the filters
    fs << "RootFilters" << rootFilters;
    fs << "RootPCAFilters" << rootPCAFilters;
    fs << "PartFilters" << partFilters;
    fs << "PartPCAFilters" << partPCAFilters;

    // write the pruning threshold
    fs << "PrunThreshold" << "[";
    for (unsigned int i = 0; i < prunThreshold.size(); i++)
        fs << prunThreshold[i];
    fs << "]";

    // write anchor points
    fs << "Anchor" << "[";
    for (unsigned int i = 0; i < anchors.size(); i++)
        fs << anchors[i];
    fs << "]";

    // write deformation
    fs << "Deformation" << "[";
    for (unsigned int i = 0; i < defs.size(); i++)
        fs << defs[i];
    fs << "]";

    // write number of parts
    fs << "NumParts" << numParts;

    // write part order
    fs << "PartOrder" << "[";
    for (unsigned int i = 0; i < partOrder.size(); i++)
        fs << partOrder[i];
    fs << "]";

    // write location weight
    fs << "LocationWeight" << "[";
    for (unsigned int i = 0; i < locationWeight.size(); i++)
        fs << locationWeight[i];
    fs << "]";

    fs.release();

    return true;
}

bool CascadeModel::deserialize(const std::string &filename)
{
    FileStorage fs;
    bool is_ok = fs.open(filename, FileStorage::READ);

    if (!is_ok) return false;

    fs["SBin"] >> sBin;
    fs["Interval"] >> interval;
    fs["MaxSizeX"] >> maxSizeX;
    fs["MaxSizeY"] >> maxSizeY;
    fs["NumComponents"] >> numComponents;
    fs["NumFeatures"] >> numFeatures;
    fs["PCADim"] >> pcaDim;
    fs["ScoreThreshold"] >> scoreThresh;
    fs["PCAcoeff"] >> pcaCoeff;
    fs["Bias"] >> bias;
    fs["RootFilters"] >> rootFilters;
    fs["RootPCAFilters"] >> rootPCAFilters;
    fs["PartFilters"] >> partFilters;
    fs["PartPCAFilters"] >> partPCAFilters;

    // read pruning threshold
    FileNode nodePrun = fs["PrunThreshold"];
    prunThreshold.resize(nodePrun.size());
    for (unsigned int i = 0; i < prunThreshold.size(); i++)
        nodePrun[i] >> prunThreshold[i];

    // read anchor points
    FileNode nodeAnchor = fs["Anchor"];
    anchors.resize(nodeAnchor.size());
    for (unsigned int i = 0; i < anchors.size(); i++)
        nodeAnchor[i] >> anchors[i];

    // read deformation
    FileNode nodeDef = fs["Deformation"];
    defs.resize(nodeDef.size());
    for (unsigned int i = 0; i < nodeDef.size(); i++)
        nodeDef[i] >> defs[i];

    // read number of parts in each component
    fs["NumParts"] >> numParts;

    // read part order
    FileNode nodeOrder = fs["PartOrder"];
    partOrder.resize(nodeOrder.size());
    for (unsigned int i = 0; i < nodeOrder.size(); i++)
        nodeOrder[i] >> partOrder[i];

    // read location weight
    FileNode nodeLoc = fs["LocationWeight"];
    locationWeight.resize(nodeLoc.size());
    for (unsigned int i = 0; i < locationWeight.size(); i++)
        nodeLoc[i] >> locationWeight[i];

    // close the file store
    fs.release();

    return true;
}

}
}
