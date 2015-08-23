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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "../precomp.hpp"
#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>

namespace cv {
namespace dnn {

#if defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
#include "THDiskFile.h"

enum LuaType
{
    TYPE_NIL      = 0,
    TYPE_NUMBER   = 1,
    TYPE_STRING   = 2,
    TYPE_TABLE    = 3,
    TYPE_TORCH    = 4,
    TYPE_BOOLEAN  = 5,
    TYPE_FUNCTION = 6,
    TYPE_RECUR_FUNCTION = 8,
    LEGACY_TYPE_RECUR_FUNCTION = 7
};

template<typename T>
static String toString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

static inline bool startsWith(const String &str, const char *substr)
{
    return str.find(substr) == 0;
}

static inline bool endsWith(const String &str, const char *substr)
{
    return str.rfind(substr) == str.length() - strlen(substr);
}

struct TorchImporter : public ::cv::dnn::Importer
{
    Net net;

    THFile *file;
    std::set<int> readedIndexes;
    std::map<int, Mat> storages;
    std::map<int, Blob> tensors;

    struct Module
    {
        String thName, apiType;
        dnn::LayerParams params;
        std::vector<Module*> modules;

        Module(const String &_thName, const String &_apiType = String())
            : thName(_thName), apiType(_apiType) {}

        ~Module()
        {
            for (size_t i = 0; i < modules.size(); i++)
                delete modules[i];
        }
    };

    Module *rootModule;
    Module *curModule;
    int moduleCounter;

    TorchImporter(String filename, bool isBinary)
    {
        rootModule = curModule = NULL;
        moduleCounter = 0;

        file = THDiskFile_new(filename.c_str(), "r", 0);
        CV_Assert(file && THFile_isOpened(file));

        if (isBinary)
            THFile_binary(file);
        else
            THFile_ascii(file);
    }

    /* Simple readers */

    inline int readInt()
    {
        return THFile_readIntScalar(file);
    }

    inline long readLong()
    {
        return THFile_readLongScalar(file);
    }

    inline bool readBool()
    {
        return readInt() != 0;
    }

    inline double readDouble()
    {
        return THFile_readDoubleScalar(file);
    }

    inline String readString()
    {
        int size = THFile_readIntScalar(file);
        String str(size, '\0');
        THFile_readCharRaw(file, const_cast<char*>(str.c_str()), size);
        return str;
    }

    inline String readTorchClassName()
    {
        String version = readString();
        return startsWith(version, "V ") ? readString() : version;
    }

    inline void readFunction()
    {
        readString();
        readObject();
    }

    void readTable(int index = -1)
    {
        index = (index < 0) ? readInt() : index;

        if (readedIndexes.count(index))
            return;

        readedIndexes.insert(index);

        int size = readInt();
        for (int i = 0; i < size; i++)
        {
            readObject(); //key
            readObject(); //value
        }
    }

    /* Special readers */

    static inline int parseTorchType(const String &str, const char *suffix, const char *prefix = "torch.")
    {
        if (startsWith(str, prefix) && endsWith(str, suffix))
        {
           String typeStr = str.substr(strlen(prefix), str.length() - strlen(prefix) - strlen(suffix));

           if (typeStr == "Double")
               return CV_64F;
           else if (typeStr == "Float")
               return CV_32F;
           else if (typeStr == "Byte")
               return CV_8U;
           else if (typeStr == "Char")
               return CV_8S;
           else if (typeStr == "Short")
               return CV_16S;
           else if (typeStr == "Int")
               return CV_32S;
           else if (typeStr == "Long") //Carefully! CV_64S type coded as CV_USRTYPE1
               return CV_USRTYPE1;
           else
               CV_Error(Error::StsNotImplemented, "Unknown type \"" + typeStr + "\" of torch class \"" + str + "\"");
        }

        return -1;
    }

    static int parseTensorType(const String &className)
    {
        return parseTorchType(className, "Tensor");
    }

    static int parseStorageType(const String &className)
    {
        return parseTorchType(className, "Storage");
    }

    void readTorchStorage(int index, int type = -1)
    {
        long size = readLong();
        Mat storageMat(1, size, (type != CV_USRTYPE1) ? type : CV_64F); //handle LongStorage as CV_64F Mat

        switch (type)
        {
        case CV_32F:
            THFile_readFloatRaw(file, (float*)storageMat.data, size);
            break;
        case CV_64F:
            THFile_readDoubleRaw(file, (double*)storageMat.data, size);
            break;
        case CV_8S:
        case CV_8U:
            THFile_readByteRaw(file, (uchar*)storageMat.data, size);
            break;
        case CV_16S:
        case CV_16U:
            THFile_readShortRaw(file, (short*)storageMat.data, size);
            break;
        case CV_32S:
            THFile_readIntRaw(file, (int*)storageMat.data, size);
            break;
        case CV_USRTYPE1:
        {
            double *buf = storageMat.ptr<double>();
            THFile_readLongRaw(file, (long*)buf, size);

            for (size_t i = (size_t)size; i-- > 0; )
                buf[i] = ((long*)buf)[i];
        }
            break;
        default:
            CV_Error(Error::StsInternal, "");
            break;
        }

        storages.insert(std::make_pair(index, storageMat));
    }

    void readTorchTable(Dict &scalarParams, std::map<String, Blob> &tensorParams)
    {
        int luaType = readInt();
        int index = readInt();

        CV_Assert(luaType == TYPE_TABLE && readedIndexes.count(index) == 0);
        readedIndexes.insert(index);

        long fpos;
        int numPairs = readInt();

        for (int i = 0; i < numPairs; i++)
        {
            fpos = THFile_position(file);
            int ktype = readInt();

            if (ktype != TYPE_STRING) //skip non-string fileds
            {
                THFile_seek(file, fpos);
                readObject(); //key
                readObject(); //value
                continue;
            }

            String key = readString();
            std::cout << i << "th key: " << key << "\n";

            fpos = THFile_position(file);
            int vtype = readInt();

            if (vtype == TYPE_TORCH)
            {
                int index = readInt();
                readTorchObject(index);

                if (tensors.count(index)) //tensor was readed
                {
                    tensorParams.insert(std::make_pair(key, tensors[index]));
                }
                else if (storages.count(index)) //storage was readed
                {
                    Mat &matStorage = storages[index];
                    Mat matCasted;
                    matStorage.convertTo(matCasted, CV_64F);

                    DictValue scalar = DictValue::arrayReal(matCasted.ptr<double>(), matCasted.total());
                    scalarParams.set(key, scalar);
                }
            }
            else if (vtype == TYPE_NUMBER)
            {
                scalarParams.set(key, readDouble());
            }
            else if (vtype == TYPE_STRING)
            {
                scalarParams.set(key, readString());
            }
            else if (vtype == TYPE_BOOLEAN)
            {
                scalarParams.set(key, readBool());
            }
            else
            {
                THFile_seek(file, fpos);
                readObject();
            }
        }

        //Debug output
        std::cout << "scalarParams:\n";
        std::cout << scalarParams;

        std::cout << "#" << tensorParams.size() << " tensorParams:\n";
        std::map<String,Blob>::const_iterator it;
        for (it = tensorParams.begin(); it != tensorParams.end(); it++)
            std::cout << it->first << ": Tensor " << it->second.shape() << "\n";
    }

    void readTorchTensor(int indexTensor, int typeTensor)
    {
        int ndims = readInt();
        AutoBuffer<long, 4> sizes(ndims);
        AutoBuffer<long, 4> steps(ndims);
        THFile_readLongRaw(file, sizes, ndims);
        THFile_readLongRaw(file, steps, ndims);
        long offset = readLong() - 1;

        //read Storage
        int typeidx = readInt();
        CV_Assert(typeidx == TYPE_TORCH || (typeidx == TYPE_NIL && ndims == 0));

        if (typeidx == TYPE_NIL)
        {
            tensors.insert(std::make_pair(indexTensor, Blob()));
            return;
        }

        int indexStorage = readInt();
        if (readedIndexes.count(indexStorage) == 0)
        {
            int typeStorage = parseStorageType(readTorchClassName());
            CV_Assert(typeStorage >= 0 && typeTensor == typeStorage);
            readTorchStorage(indexStorage, typeStorage);
        }

        //small check
        size_t requireElems = (size_t)offset + (size_t)steps[0] * (size_t)sizes[0];
        size_t storageElems = storages[indexStorage].total();
        if (requireElems > storageElems)
            CV_Error(Error::StsBadSize, "Storage has insufficent number of elemements for requested Tensor");

        //convert sizes
        AutoBuffer<int, 4> isizes(ndims);
        AutoBuffer<size_t, 4> ssteps(ndims);
        for (int i = ndims - 1; i >= 0; i--)
        {
            isizes[i] = (int)sizes[i];
            ssteps[i] = (size_t)steps[i] * CV_ELEM_SIZE(typeTensor);
        }

        //allocate Blob
        Mat srcMat(ndims, (int*)isizes, typeTensor , storages[indexStorage].ptr() + offset, (size_t*)ssteps);
        //int dstType = (typeTensor == CV_64F) ? CV_64F : CV_32F;
        int dstType = CV_32F;

        Blob blob;
        blob.create(BlobShape(ndims, isizes), dstType);
        srcMat.convertTo(blob.matRef(), dstType);

        tensors.insert(std::make_pair(indexTensor, blob));
    }

    static bool isNNClass(const String &className, String &nnName)
    {
        const char *prefixes[] = {"nn.", "cunn.", "cudnn.", "fbcunn.", NULL};

        for (int i = 0; prefixes[i]; i++)
        {
            if (startsWith(className, prefixes[i]))
            {
                nnName = className.substr(strlen(prefixes[i]));
                return true;
            }
        }

        return false;
    }

    static void convertTorchKernelsParams(const Dict &torchParams, cv::dnn::LayerParams &layerParams)
    {
        layerParams.set("kernel_h", torchParams.get<int>("kH"));
        layerParams.set("kernel_w", torchParams.get<int>("kW"));
        layerParams.set("stride_h", torchParams.get<int>("dH"));
        layerParams.set("stride_w", torchParams.get<int>("dW"));
        layerParams.set("pad_h", torchParams.get<int>("padH", 0));
        layerParams.set("pad_w", torchParams.get<int>("padW", 0));
    }

    void readTorchObject(int index)
    {
        if(readedIndexes.count(index))
        {
            if(!storages.count(index) && !tensors.count(index))
                CV_Error(Error::StsNotImplemented, "Objects which have multiple references are not supported");
            else
                return;
        }

        String className = readTorchClassName();
        String nnName;
        std::cout << "Class: " << className << std::endl;

        int type;
        if ( (type = parseTensorType(className)) >= 0 ) //is Tensor
        {
            readTorchTensor(index, type);
        }
        else if ( (type = parseStorageType(className)) >= 0 ) //is Storage
        {
            readTorchStorage(index, type);
        }
        else if (isNNClass(className, nnName))
        {
            Dict scalarParams;
            std::map<String, Blob> tensorParams;

            Module *newModule = new Module(nnName);
            cv::dnn::LayerParams &layerParams = newModule->params;

            if (nnName == "Sequential" || nnName == "Parallel" || nnName == "Concat")
            {
                Module *parentModule = curModule;
                curModule->modules.push_back(newModule);
                curModule = newModule;
                readTorchTable(scalarParams, tensorParams);
                curModule = parentModule;

                if (nnName == "Parallel")
                {
                    layerParams.set("inputDimension", scalarParams.get<int>("inputDimension"));
                    layerParams.set("outputDimension", scalarParams.get<int>("outputDimension"));
                }
                if (nnName == "Concat")
                {
                    layerParams.set("dimension", scalarParams.get<int>("dimension"));
                }
            }
            else if (nnName == "SpatialConvolution")
            {
                newModule->apiType = "Convolution";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(tensorParams.count("weight"));
                layerParams.blobs.push_back(tensorParams["weight"]);

                bool bias = tensorParams.count("bias") != 0;
                layerParams.set("bias_term", bias);
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"]);

                layerParams.set("num_output", scalarParams.get<int>("nOutputPlane"));
                convertTorchKernelsParams(scalarParams, layerParams);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialMaxPooling" || nnName == "SpatialAveragePooling")
            {
                newModule->apiType = "Pooling";
                readTorchTable(scalarParams, tensorParams);

                if (nnName == "SpatialMaxPooling")
                    layerParams.set("pool", "MAX");
                if (nnName == "SpatialAveragePooling")
                    layerParams.set("pool", "AVE");
                convertTorchKernelsParams(scalarParams, layerParams);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Linear")
            {
                newModule->apiType = "InnerProduct";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(tensorParams.count("weight"));
                Blob weightBlob = tensorParams["weight"];
                layerParams.blobs.push_back(weightBlob);

                bool bias = tensorParams.count("bias") != 0;
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"]);
                layerParams.set("bias_term", bias);

                layerParams.set("num_output", weightBlob.size(0));
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Reshape")
            {
                newModule->apiType = "Reshape";

                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("size"));

                DictValue dimParam = scalarParams.get("size");
                layerParams.set("dim", dimParam);

                if (scalarParams.has("batchMode") && scalarParams.get<bool>("batchMode"))
                    layerParams.set("axis", 1);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "ReLU")
            {
                curModule->modules.push_back(new Module(nnName, "ReLU"));
                readObject();
            }
            else if (nnName == "Tanh")
            {
                curModule->modules.push_back(new Module(nnName, "TanH"));
                readObject();
            }
            else if (nnName == "Sigmoid")
            {
                curModule->modules.push_back(new Module(nnName, "Sigmoid"));
                readObject();
            }
            else
            {
                delete newModule;
                CV_Error(Error::StsNotImplemented, "Unknown nn class \"" + className + "\"");
                readObject();
            }
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported Torch class \"" + className + "\"");
        }

        readedIndexes.insert(index);
    }

    void readObject()
    {
        int typeidx = readInt();

        if (typeidx == TYPE_TORCH)
        {
            int index = readInt();
            readTorchObject(index);
            readedIndexes.insert(index);
        }
        else if (typeidx == TYPE_NIL)
            return;
        else if (typeidx == TYPE_NUMBER)
            readDouble();
        else if (typeidx == TYPE_BOOLEAN)
            readBool();
        else if (typeidx == TYPE_STRING)
            readString();
        else if (typeidx == TYPE_TABLE)
            readTable();
        else
            CV_Error(Error::StsNotImplemented, "Unsupported Lua type");
    }

    inline String generateLayerName(const String &label = String())
    {
        return "l" + toString(++this->moduleCounter) + "_" + label;
    }

    int fill(Module *module, int prevLayerId = 0, int prevOutNum = 0)
    {
        if (module == NULL)
            return prevLayerId;

        if (module->apiType.length())
        {
            int newLayerId = this->net.addLayer(generateLayerName(module->apiType), module->apiType, module->params);
            net.connect(prevLayerId, prevOutNum, newLayerId, 0);
            return newLayerId;
        }
        else
        {
            if (module->thName == "Sequential")
            {
                for (size_t i = 0; i < module->modules.size(); i++)
                {
                    prevLayerId = fill(module->modules[i], prevLayerId, prevOutNum);
                    prevOutNum = 0;
                }
                return prevLayerId;
            }
            else if (module->thName == "Concat")
            {
                int newId, splitId, mergeId;
                LayerParams mergeParams, splitParams;
                mergeParams.set("axis", module->params.get<int>("dimension") - 1);

                splitId = net.addLayer(generateLayerName("torchSplit"), "Split", splitParams);
                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);
                net.connect(prevLayerId, prevOutNum, splitId, 0);

                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    newId = fill(module->modules[i], splitId, i);
                    net.connect(newId, 0, mergeId, i);
                }

                return mergeId;
            }
            else if (module->thName == "Parallel")
            {
                int newId, splitId, mergeId, reshapeId;

                LayerParams splitParams, mergeParams, reshapeParams;
                splitParams.set("axis", module->params.get<int>("inputDimension") - 1);
                mergeParams.set("axis", module->params.get<int>("outputDimension") - 1);
                reshapeParams.set("axis", splitParams.get<int>("axis"));
                reshapeParams.set("num_axes", 1);

                splitId = net.addLayer(generateLayerName("torchSplit"), "Slice", splitParams);
                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);
                reshapeId = net.addLayer(generateLayerName("torchReshape"), "Reshape", reshapeParams);
                net.connect(prevLayerId, prevOutNum, splitId, 0);

                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    net.connect(splitId, i, reshapeId, i);
                    newId = fill(module->modules[i], reshapeId, i);
                    net.connect(newId, 0, mergeId, i);
                }

                return mergeId;
            }
        }

        CV_Error(Error::StsInternal, "Unexpected torch container: " + module->thName);
        return -1;
    }

    void populateNet(Net net)
    {
        if (rootModule == NULL)
        {
            rootModule = new Module("Sequential");
            curModule = rootModule;

            THFile_seek(file, 0);
            readObject();
        }

        this->net = net;
        fill(rootModule);
    }
};

CV_EXPORTS Ptr<Importer> createTorchImporter(const String &filename, bool isBinary)
{
    return Ptr<Importer>(new TorchImporter(filename, isBinary));
}


CV_EXPORTS Blob readTorchBlob(const String &filename, bool isBinary)
{
    Ptr<TorchImporter> importer(new TorchImporter(filename, isBinary));
    importer->readObject();
    CV_Assert(importer->tensors.size() == 1);

    return importer->tensors.begin()->second;
}

#else //ENABLE_TORCH_IMPORTER

CV_EXPORTS Ptr<Importer> createTorchImporter(const String&, bool)
{
    CV_Error(Error::StsNotImplemented, "Module was build without Torch importer");
    return Ptr<Importer>();
}

CV_EXPORTS Blob readTorchMat(const String&, bool)
{
    CV_Error(Error::StsNotImplemented, "Module was build without Torch importer");
    return Blob();
}

#endif //ENABLE_TORCH_IMPORTER
}
}
