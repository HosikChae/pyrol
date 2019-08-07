//
// Created by jeff on 12/12/18.
//

#ifndef VNDATACOLLECT_CSVWRITER_H
#define VNDATACOLLECT_CSVWRITER_H
#include <iostream>
#include <fstream>
#include <vn/sensors.h>

using namespace std;
using namespace vn::math;

class CSVWriter
{
    std::string fileName;
    std::string delimeter;
    int linesCount;
    ofstream imuDataFile;
public:
    // Constructor
    CSVWriter(string filename, string delm = ",") :
            fileName(filename), delimeter(delm), linesCount(0)
    {}
    // Member function for adding a new row of IMU data;
    void addRowData(double sysTime, uint64_t startupTime, vec3f rates, vec3f accels);
    // Member function for opening file
    void openFile();
    // Member function for closing file
    void closeFile();
};
#endif //VNDATACOLLECT_CSVWRITER_H
