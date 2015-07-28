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
        readObject(true);
    }

    void readTable()
    {
        std::cout << "Skipping table\n";

        int index = readInt();
        CV_Assert(readedIndexes.count(index) == 0);
        readedIndexes.insert(index);

        int size = readInt();
        for (int i = 0; i < size; i++)
        {
            readObject(true); //key
            readObject(true); //value
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
        long size = readLong();
        Mat storageMat(1, size, type);

        THFile_readByteRaw(file, storageMat.data, size * CV_ELEM_SIZE(type));

        storages.insert(std::make_pair(index, storageMat));
        readedIndexes.insert(index);
    }

    Blob readTorchTensor(int typeTensor, bool skip = false)
    {
        int ndims = readInt();

        AutoBuffer<long, 4> sizes(ndims);
        AutoBuffer<long, 4> steps(ndims);
        THFile_readLongRaw(file, sizes, ndims);
        THFile_readLongRaw(file, sizes, ndims);

        long offset = readLong() - 1;

        //read Storage
        int typeidx = readInt();
        std::cout << "stograge typeidx of tensor: " << typeidx << "\n";
        CV_Assert(typeidx == TYPE_TORCH || (typeidx == TYPE_NIL && ndims == 0));

        if (typeidx == TYPE_NIL)
            return Blob();

        int index = readInt();
        if (readedIndexes.count(index) == 0)
        {
            int typeStorage = parseStorageType(readTorchClassName());
            CV_Assert(typeStorage >= 0 && typeTensor == typeStorage);
            readTorchStorage(typeStorage, index);
        }

        //allocate Blob
        AutoBuffer<int, 4> isizes(ndims);
        AutoBuffer<size_t, 4> ssteps(ndims);

        size_t stepExpected = 1;
        for (int i = ndims - 1; i >= 0; i--)
        {
            isizes[i] = (int)sizes[i];
            ssteps[i] = (size_t)steps[i] * CV_ELEM_SIZE(typeTensor);

            stepExpected *= sizes[i];
        }

        if (skip)
            return Blob();

        Mat srcMat(ndims, (int*)isizes, typeTensor , storages[index].ptr(), (size_t*)ssteps);
        int dstType = (typeTensor == CV_64F) ? CV_64F : CV_32F;

        Blob blob;
        blob.create(BlobShape(ndims, isizes), dstType);
        srcMat.convertTo(blob.getMatRef(), dstType);

        return blob;
    }

    void readTorchObject(int index, bool skip = false)
    {
        String className = readTorchClassName();
        std::cout << "Class: " << className << std::endl;

        int type;
        if ( (type = parseTensorType(className)) >= 0 ) //is Tensor
        {
            readTorchTensor(type);
            return;
        }
        else if ( (type = parseStorageType(className)) >= 0 ) //is Storage
        {
            readTorchStorage(index, type);
        }
        else if (className == "nn.Sequential")
        {
            readObject();
        }
        else if (className == "nn.Concat")
        {
            readObject();
        }
        else if (className == "nn.SpatialConvolution")
        {
            readObject();
        }
        else if (className == "nn.ReLU")
        {
            readObject();
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported Torch class \"" + className +"\"");
        }
    }

    void readObject(bool skip = false)
    {
        int typeidx = readInt();
        std::cout << "typeidx: " << typeidx << "\n";

        if (typeidx == TYPE_TORCH)
        {
            int index = readInt();

            if (readedIndexes.count(index) == 0)
            {
                readTorchObject(index, skip);
                readedIndexes.insert(index);
            }
            else
            {
                //CV_Error(Error::StsNotImplemented, "");
                //TBD
            }
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

    void populateNet(Net net)
    {
        THFile_seek(file, 0);
        readedIndexes.clear();

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
