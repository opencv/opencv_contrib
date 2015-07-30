#include "../precomp.hpp"
#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>

namespace cv {
namespace dnn {

#if ENABLE_TORCH_IMPORTER || 1
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
    THFile *file;
    std::set<int> readedIndexes;
    std::map<int, Mat> storages;
    std::map<int, Blob> tensors;

    TorchImporter(String filename, bool isBinary)
    {
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
        return readInt();
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
        if (storages.count(index))
            return;

        long size = readLong();
        Mat storageMat(1, size, type);

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
        default:
            CV_Error(Error::StsInternal, "");
            break;
        }

        storages.insert(std::make_pair(index, storageMat));
        readedIndexes.insert(index);
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
                readObject();
                readObject();
                continue;
            }

            String key = readString();
            
            fpos = THFile_position(file);
            int vtype = readInt();

            if (vtype == TYPE_TORCH)
            {
                int index = readInt();
                if (tensors.count(index) == 0)
                {
                    readTorchObject(index);
                }
                
                if (tensors.count(index))
                    tensorParams.insert(std::make_pair(key, tensors[index]));
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
                continue;
            }
        }
    }

    void readTorchTensor(int indexTensor, int typeTensor)
    {
        if (tensors.count(indexTensor))
            return;

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
        int dstType = (typeTensor == CV_64F) ? CV_64F : CV_32F;

        Blob blob;
        blob.create(BlobShape(ndims, isizes), dstType);
        srcMat.convertTo(blob.getMatRef(), dstType);

        tensors.insert(std::make_pair(indexTensor, blob));
    }

    bool isNNClass(const String &className, String &nnName)
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

    void readTorchObject(int index, bool skip = false)
    {
        String className = readTorchClassName();
        String nnName;
        std::cout << "Class: " << className << std::endl;

        Dict scalarParams;
        std::map<String, Blob> tensorParams;

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
            CV_Assert(!readedIndexes.count(index));
            readTorchTable(scalarParams, tensorParams);

            std::cout << scalarParams;
            for (std::map<String,Blob>::const_iterator it = tensorParams.begin(); it != tensorParams.end(); it++)
                std::cout << it->first << ": Tensor" << "\n";

            if (nnName == "Sequential")
            {
            }
            else if (nnName == "Concat")
            {
            }
            else if (nnName == "SpatialConvolution")
            {
            }
            else if (nnName == "ReLU")
            {
            }
            else
            {
                CV_Error(Error::StsNotImplemented, "Unknown nn class \"" + className + "\"");
            }
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported Torch class \"" + className + "\"");
        }
    }

    void readObject()
    {
        int typeidx = readInt();
        std::cout << "typeidx: " << typeidx << "\n";

        if (typeidx == TYPE_TORCH)
        {
            int index = readInt();

            if (readedIndexes.count(index) == 0)
            {
                readTorchObject(index);
                readedIndexes.insert(index);
            }
        }
        else if (typeidx == TYPE_NIL)
            return;
        else if (typeidx == TYPE_NUMBER)
            //readDouble();
            std::cout << readDouble() << std::endl;
        else if (typeidx == TYPE_BOOLEAN)
            readBool();
        else if (typeidx == TYPE_STRING)
            //readString();
            std::cout << readString() << std::endl;
        else if (typeidx == TYPE_TABLE)
            readTable();
        else
            CV_Error(Error::StsNotImplemented, "Unsupported Lua type");
    }

    void populateNet(Net net)
    {
        THFile_seek(file, 0);
        readedIndexes.clear();
        storages.clear();

        readObject();
    }
};

CV_EXPORTS Ptr<Importer> createTorchImporter(const String &filename, bool isBinary)
{
    return Ptr<Importer>(new TorchImporter(filename, isBinary));
}

#else //ENABLE_TORCH_IMPORTER

CV_EXPORTS Ptr<Importer> createTorchImporter(const String&, bool)
{
    CV_Error(Error::StsNotImplemented, "Module was build without Torch importer");
    return Ptr<Importer>();
}

#endif //ENABLE_TORCH_IMPORTER
}
}
