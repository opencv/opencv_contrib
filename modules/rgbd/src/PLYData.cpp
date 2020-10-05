//
//  DataImporter.cpp
//  Used to import data from PLY file to OpenCV mat format (as a point cloud for now)
//  Can be easily modified (using the complete header parsing done here) to import to other
//  formats (OpenCV or others).
//
//  Created by Cedric Leblond Menard on 16-07-26.
//  Copyright © 2016 Cedric Leblond Menard. All rights reserved.
//
#include "precomp.hpp"
#include "plydata.hpp"
#include <iomanip>

namespace cv
{
namespace pcseg
{

#define DEBUG_MODE 1

// MARK: Supporting variables and functions
// Dictionary for file format
const char* enum2str[] = {
    "ascii",
    "binary_big_endian",
    "binary_little_endian"
};

// Function to reverse float endianness
float ReverseFloat( const float inFloat )
{
    float retVal;
    char *floatToConvert = ( char* ) & inFloat;
    char *returnFloat = ( char* ) & retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

// Check system endianness
bool isLittleEndian()
{
    uint16_t number = 0x1;
    char *numPtr = (char*)&number;
    return (numPtr[0] == 1);
}

// Getline wether it is /r/n or simply /n ended
// From : http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
std::istream& safeGetline(std::istream& is, std::string& t)
{
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
            case '\n':
                return is;
            case '\r':
                if(sb->sgetc() == '\n')
                    sb->sbumpc();
                return is;
            case EOF:
                // Also handle the case when the last line has no line ending
                if(t.empty())
                    is.setstate(std::ios::eofbit);
                return is;
            default:
                t += (char)c;
        }
    }
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

// MARK: DataImporter
DataImporter::DataImporter(cv::Mat &outputData, cv::Mat &outputColor, std::string inputFile):
data(outputData), colors(outputColor), filename(inputFile)
{
    // Get data from header of file
    getHeader();


    // Open file stream and seek to data position
    switch (header.format) {
        case PLY_ASCII:
            filestream.open(filename,std::ios::in);
            break;
        case PLY_BIN_BIGEND:
            filestream.open(filename,std::ios::in | std::ios::binary);
            break;
        case PLY_BIN_LITEND:
            filestream.open(filename,std::ios::in | std::ios::binary);
            break;
    }
    filestream.seekg(dataPosition);
}

DataImporter::~DataImporter() {
    filestream.close();
}

void DataImporter::getHeader() {
    filestream.open(filename,std::ios::in);
    std::string line;
    safeGetline(filestream, line);
    if (line != "ply") {
        fileIsValid = false;
#ifdef DEBUG_MODE
        assert(false);
#endif
        return;
    } else {
        fileIsValid = true;
    }

    safeGetline(filestream,line);
    std::vector<std::string> parsed = split(line, ' ');
    while (line != "end_header") {

        // Parse format line
        if (parsed[0].compare("format") == 0) {
            if (parsed[1] == enum2str[PLY_ASCII]) {
                header.format = PLY_ASCII;
            } else if (parsed[1] == enum2str[PLY_BIN_BIGEND]) {
                header.format = PLY_BIN_BIGEND;
            } else if (parsed[1] == enum2str[PLY_BIN_LITEND]) {
                header.format = PLY_BIN_LITEND;
            } else {
                fileIsValid = false;
#ifdef DEBUG_MODE
                assert(false);
#endif
                return;
            }
            safeGetline(filestream,line);
            parsed = split(line, ' ');

        // Parse comment line
        } else if (parsed[0].compare("comment") == 0) {
            header.comment = line.substr(8);
            safeGetline(filestream,line);
            parsed = split(line, ' ');

        // Parse element line
        } else if (parsed[0].compare("element") == 0) {
            PLYElement elem;
            if (parsed[1].compare("vertex") == 0) {
                elem.type = PLYVertex;
            } else if (parsed[1].compare("face") == 0) {
                elem.type = PLYFace;
            } else {
                fileIsValid = false;
#ifdef DEBUG_MODE
                assert(false);
#endif
                return;
            }
            elem.numberOfElements = atoi(parsed[2].c_str());    // Number of elements

            // Parse through all properties
            safeGetline(filestream,line);
            std::vector<std::string> parsed = split(line, ' ');
            uint typeIdx = 1;

            while (parsed[0].compare("property") == 0) {
                PLYProperty property;
                if (parsed[typeIdx].compare("list") == 0) {
                    typeIdx += 2;
                    property.isList = true;
                } else {
                    property.isList = false;
                }
                if (parsed[typeIdx].compare("char") == 0) {
                    property.type = CHAR;
                } else if (parsed[typeIdx].compare("uchar") == 0) {
                    property.type = UCHAR;
                } else if (parsed[typeIdx].compare("int") == 0) {
                    property.type = INT;
                } else if (parsed[typeIdx].compare("uint") == 0) {
                    property.type = UINT;
                } else if (parsed[typeIdx].compare("float") == 0) {
                    property.type = FLOAT;
                } else if (parsed[typeIdx].compare("double") == 0) {
                    property.type = DOUBLE;
                } else {
                    fileIsValid = false;
#ifdef DEBUG_MODE
                    assert(false);
#endif
                    return;
                }
                property.name = parsed[typeIdx+1];
                elem.properties.emplace_back(property);

                // Get next line
                safeGetline(filestream,line);
                parsed = split(line, ' ');
            }
            header.elements.emplace_back(elem);

        // Unknown header data
        } else {
            fileIsValid = false;
#ifdef DEBUG_MODE
            assert(false);
#endif
            return;
        }
    }
    dataPosition = filestream.tellg();
    filestream.close();

}

void DataImporter::importPCDataFromFile() {
    // Initialize data
    //PCData data;
    std::string line;


    // TODO: Add file header verification (check if data is as per good ply format)

    // Prepare matrices
    data = cv::Mat_<float>(header.elements[0].numberOfElements,3);
    colors = cv::Mat_<unsigned char>(header.elements[0].numberOfElements,3);

    float* points_ptr = (float*)data.data;
    uchar* colors_ptr = colors.data;

    // Start parsing data
    switch (header.format) {
        case PLY_ASCII:
            for (uint i = 0; i < header.elements[0].numberOfElements; i++) {
                safeGetline(filestream, line);
                std::vector<std::string> parsed = split(line, ' ');

                *points_ptr = stof(parsed[0]);
                points_ptr[1] = stof(parsed[1]);
                points_ptr[2] = stof(parsed[2]);

                // Inverted format (OpenCV uses BGR)
                *colors_ptr = stof(parsed[5]);
                colors_ptr[1] = stof(parsed[4]);
                colors_ptr[2] = stof(parsed[3]);
                points_ptr += 3;
                colors_ptr += 3;
            }
            break;
        case PLY_BIN_BIGEND:    // Fallthrough, we deal with both
        case PLY_BIN_LITEND:
            if ( ( (format == PLY_BIN_LITEND) && isLittleEndian() ) ||
                ( (format == PLY_BIN_BIGEND) && !isLittleEndian() )
                ) {
                // If same endianness, just parse
                for (uint i = 0; i < header.elements[0].numberOfElements; i++) {
                    char tempFloat[sizeof(float)];
                    for (uint j = 0; j < 3; j++) {
                        filestream.read(tempFloat, sizeof(float));
                        points_ptr[j] = reinterpret_cast<float&>(tempFloat);
                    }

                    char temp[1];
                    // Inverted format (OpenCV uses BGR)
                    for (int j = 2; j > 0; j--) {
                        filestream.read(temp, 1);
                        colors_ptr[j] = temp[0];
                    }
                    points_ptr += 3;
                    colors_ptr += 3;
                }
            } else {
                // If endianness of system != file endianness, reverse order
                for (uint i = 0; i < header.elements[0].numberOfElements; i++) {
                    char tempFloat[sizeof(float)];
                    for (uint j = 0; j < 3; j++) {
                        filestream.read(tempFloat, sizeof(float));
                        points_ptr[j] = ReverseFloat(reinterpret_cast<float&>(tempFloat));  // Reverse
                    }

                    char temp[1];
                    // Inverted format (OpenCV uses BGR)
                    for (int j = 2; j > 0; j--) {
                        filestream.read(temp, 1);
                        colors_ptr[j] = temp[0];
                    }
                    points_ptr += 3;
                    colors_ptr += 3;
                }
            }
            break;
    }
}

bool DataImporter::isFileValid() {
    return fileIsValid;
}

// MARK: DataExporter
DataExporter::DataExporter(cv::Mat outputData, cv::Mat outputColor, std::string outputfile, FileFormat outputformat) :
filename(outputfile), format(outputformat), data(outputData)
{
    // MARK: Init
    // Make color conversion if necessary
    if (outputColor.channels() == 3)
    {
        colors = outputColor;
    } else if (outputColor.cols == 3 && outputColor.channels() == 1) {
        // If is matrix of Nx3 for colors, reshape to Nx1 3 channel matrix
        colors = outputColor.reshape(3);
    } else {
        cv::cvtColor(outputColor, colors, cv::COLOR_GRAY2BGR);
    }

    // Opening filestream
    switch (format) {
        case FileFormat::PLY_BIN_LITEND:
        case FileFormat::PLY_BIN_BIGEND:
            filestream.open(filename, std::ios::out | std::ios::binary);
            break;
        case FileFormat::PLY_ASCII:
            filestream.open(filename,std::ios::out);
            break;
    }


    // Calculating number of elements
    CV_Assert(data.rows == colors.rows);      // If not same size, assert

    // Check for valid matrix format
    CV_Assert(
              ((data.cols == 3 && data.type() == CV_32FC1) || (data.cols == 1 && data.type() == CV_32FC3)) &&
              ((colors.cols == 3 && colors.type() == CV_8UC1) || (colors.cols == 1 && colors.type() == CV_8UC3))
              );

    CV_Assert(data.isContinuous() &&
              colors.isContinuous());               // If not continuous in memory

    if (data.channels() == 3) {
        numElem = data.rows * data.cols;
    } else {
        numElem = data.rows;
    }

}

void DataExporter::exportToFile() {
    // MARK: Header writing
    filestream << "ply" << std::endl <<
    "format " << enum2str[format] << " 1.0" << std::endl <<
    "comment file created using code by Cedric Menard" << std::endl <<
    "element vertex " << numElem << std::endl <<
    "property float x" << std::endl <<
    "property float y" << std::endl <<
    "property float z" << std::endl <<
    "property uchar red" << std::endl <<
    "property uchar green" << std::endl <<
    "property uchar blue" << std::endl <<
    "end_header" << std::endl;

    // MARK: Data writing

    // Pointer to data
    const float* pData = data.ptr<float>(0);
    const unsigned char* pColor = colors.ptr<unsigned char>(0);
    const unsigned long numIter = 3*numElem;                            // Number of iteration (3 channels * numElem)
    const bool hostIsLittleEndian = isLittleEndian();

    float_t bufferXYZ;                                                 // Coordinate buffer for float type

    // Output format switch
    switch (format) {
        case FileFormat::PLY_BIN_BIGEND:
            // Looping through all
            for (unsigned long i = 0; i<numIter; i+=3) {                                // Loop through all elements
                for (unsigned int j = 0; j<3; j++) {                                    // Loop through 3 coordinates
                    if (hostIsLittleEndian) {
                        bufferXYZ = ReverseFloat(pData[i+j]);                        // Convert from host to network (Big endian)
                        filestream.write(reinterpret_cast<const char *>(&bufferXYZ),    // Non compiled cast to char array
                                         sizeof(bufferXYZ));
                    } else {
                        bufferXYZ = pData[i+j];
                        filestream.write(reinterpret_cast<const char *>(&bufferXYZ),    // Non compiled cast to char array
                                         sizeof(bufferXYZ));
                    }
                }
                for (int j = 2; j>=0; j--) {
                    // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                    filestream.put(pColor[i+j]);                                        // Loop through RGB
                }
            }

            break;

        case FileFormat::PLY_BIN_LITEND:                                                // Assume host as little-endian
            for (unsigned long i = 0; i<numIter; i+=3) {                                // Loop through all elements
                for (unsigned int j = 0; j<3; j++) {                                    // Loop through 3 coordinates
                    if (hostIsLittleEndian) {
                        filestream.write(reinterpret_cast<const char *>(pData+i+j),     // Non compiled cast to char array
                                         sizeof(bufferXYZ));
                    } else {
                        bufferXYZ = ReverseFloat(pData[i+j]);
                        filestream.write(reinterpret_cast<const char *>(&bufferXYZ), sizeof(bufferXYZ));
                    }
                }
                for (int j = 2; j>=0; j--) {
                    // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                    filestream.put(pColor[i+j]);                                        // Loop through RGB
                }
            }

            break;

        case FileFormat::PLY_ASCII:
            for (unsigned long i = 0; i<numIter; i+=3) {                            // Loop through all elements
                for (unsigned int j = 0; j<3; j++) {                                // Loop through 3 coordinates
                    filestream << std::setprecision(9) << pData[i+j] << " ";
                }
                for (int j = 2; j>=0; j--) {
                    // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                    filestream << (unsigned short)pColor[i+j] << (j==0?"":" ");                     // Loop through RGB
                }
                filestream << std::endl;                                            // End if element line
            }
            break;

        default:
            break;
    }
}

DataExporter::~DataExporter() {
    filestream.close();
}

}}