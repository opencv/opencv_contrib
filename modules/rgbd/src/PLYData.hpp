//
//  DataImporter.hpp
//
//
//  Created by Cedric Leblond Menard on 16-07-26.
//  Copyright © 2016 Cedric Leblond Menard. All rights reserved.
//

#ifndef plydata_hpp
#define plydata_hpp

#include "precomp.hpp"
#include <fstream>
#include <cstring>

namespace cv
{
namespace pcseg
{

enum FileFormat {PLY_ASCII, PLY_BIN_BIGEND, PLY_BIN_LITEND};
enum PLYElementType {PLYVertex, PLYFace};
enum PLYValueType {CHAR, UCHAR, SHORT, USHORT, INT, UINT, FLOAT, DOUBLE};

struct PLYProperty {
    PLYValueType type;
    std::string name;
    bool isList = false;
};

struct PLYElement {
    PLYElementType type;
    std::vector<PLYProperty> properties;
    uint numberOfElements;
};

struct PLYHeaderData {
    FileFormat format;
    std::string comment;
    std::vector<PLYElement> elements;
    float version;
};

struct PCData {
    cv::Mat points;
    cv::Mat colors;
};

class DataImporter {
private:

    // Variables
    std::string filename = "";
    FileFormat format;
    cv::Mat &data;
    cv::Mat &colors;
    std::ifstream filestream;
    unsigned long numElem;
    bool fileIsValid = true;
    long long dataPosition;

    // Header Variables
    PLYHeaderData header;

    // Methods
    void getHeader();

public:
    DataImporter(cv::Mat &outputData, cv::Mat &outputColor, std::string inputFile);
    ~DataImporter();
    void importPCDataFromFile();
    bool isFileValid();
};

class DataExporter  {
private:
    std::string filename = "";
    FileFormat format;
    cv::Mat data;
    cv::Mat colors;
    std::ofstream filestream;
    unsigned long numElem;

public:
    DataExporter(cv::Mat outputData, cv::Mat outputColor, std::string outputfile, FileFormat outputformat);
    ~DataExporter();
    void exportToFile();

};

}
}

#endif /* plydata_hpp */